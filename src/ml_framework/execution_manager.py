"""
ExecutionManager Skeleton for the ML Framework.

Acts as the central coordination facade orchestrating the execution lifecycle:
environment validation, model loading, input processing, inference execution,
output processing, telemetry tracking, and resource cleanup.

Fulfills Task 1 requirements: strict lifecycle state validation, thread safety,
timeout enforcement, safe model reload with rollback, and isolated cleanup hooks.
"""

import os
import time
import copy
import logging
import threading
import concurrent.futures
from typing import Any, Dict, Optional, Callable, Union, List

from .config.model_config import FrameworkConfig
from .results.execution_result import ExecutionResult, ExecutionContext, ExecutionState
from .errors.exceptions import (
    MLFrameworkError,
    ConfigurationError,
    StateError,
    ExecutionError,
    ExecutionTimeoutError,
    InferenceError,
    ModelLoadError,
    ResourceCleanupError,
)
from .interfaces.base import (
    BaseEnvironmentManager,
    BaseModelLoader,
    BaseInputProcessor,
    BaseInferenceEngine,
    BaseOutputProcessor,
    BaseStatusTracker,
    BaseResourceManager,
)

logger = logging.getLogger("ml_framework.ExecutionManager")


class ExecutionManager:
    """
    Central orchestration coordinator for ML execution pipelines.

    Enforces the Successful Execution Sequence, Execution Flow, and Error Handling Sequence
    without hardcoding specific model weights, application runtimes, or external network calls.

    Concurrency Policy:
    - Data plane (execute()): Supports concurrent execution across threads. Lengthy stages
      (input processing and output processing) run concurrently outside locks.
    - Inference safety: Non-reentrant model instances and inference engines are protected
      via an internal _inference_lock to serialize forward passes. The lifecycle lock (_lock)
      is NOT held during inference. Context, metrics, and results remain strictly isolated.
    - In-flight tracking: Active pipeline executions are tracked via an in-flight counter.
    - Control plane (initialize(), reload(), unload(), shutdown()): Mutations acquire the lifecycle
      lock and enforce draining of in-flight executions before altering or destroying active models.
      Reloading transitions to RELOADING, and drain barriers reject new execution entries.
    """

    def __init__(
        self,
        config: Optional[Union[FrameworkConfig, Dict[str, Any]]] = None,
        env_manager: Optional[BaseEnvironmentManager] = None,
        model_loader: Optional[BaseModelLoader] = None,
        input_processor: Optional[BaseInputProcessor] = None,
        inference_engine: Optional[BaseInferenceEngine] = None,
        output_processor: Optional[BaseOutputProcessor] = None,
        status_tracker: Optional[BaseStatusTracker] = None,
        resource_manager: Optional[BaseResourceManager] = None,
    ):
        # 1. Parse and validate configuration
        if config is None:
            self._config = FrameworkConfig()
        elif isinstance(config, dict):
            self._config = FrameworkConfig.from_dict(config)
        elif isinstance(config, FrameworkConfig):
            self._config = config
            self._config.validate()
        else:
            raise ConfigurationError(
                f"Expected FrameworkConfig or dict, got {type(config).__name__}"
            )

        # 2. Injected component contracts (Task 2 - Task 8)
        from .environment.environment_manager import EnvironmentManager
        from .input.processor import InputProcessor

        self.env_manager = env_manager if env_manager is not None else EnvironmentManager()
        self.input_processor = input_processor if input_processor is not None else InputProcessor(
            config=self._config.input_config,
            storage_config=self._config.storage_config,
        )
        self.model_loader = model_loader
        if inference_engine is not None:
            self.inference_engine = inference_engine
        elif (
            self._config.triton_config
            and self._config.triton_config.endpoint_url
            and (self._config.model_name or self._config.triton_config.model_name)
        ):
            from .inference.triton_engine import TritonInferenceEngine
            self.inference_engine = TritonInferenceEngine(config=self._config.triton_config)
        else:
            self.inference_engine = None
        from .output.processor import OutputProcessor
        from .status.tracker import StatusTracker

        self.output_processor = output_processor if output_processor is not None else OutputProcessor(
            task=self._config.task
        )
        self.status_tracker = status_tracker if status_tracker is not None else StatusTracker()
        self.resource_manager = resource_manager

        # 3. Concurrency primitives and in-flight tracking
        self._lock = threading.RLock()
        self._drain_cond = threading.Condition(self._lock)
        self._inference_lock = threading.Lock()
        self._active_executions = 0
        self._draining = False
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

        # 4. Lifecycle state and metrics
        self._state = ExecutionState.UNINITIALIZED
        self._active_model_instance = None
        self._active_model_id = (
            f"{self._config.model_name}:{self._config.model_version}"
            if self._config.model_name and self._config.model_version
            else self._config.model_source or "unspecified"
        )
        self._status_callbacks: List[Callable[[ExecutionResult], None]] = []
        self._metrics = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_duration_ms": 0.0,
            "total_duration_ms": 0.0,
        }

    # -------------------------------------------------------------------------
    # Properties (Thread-safe defensive accessors)
    # -------------------------------------------------------------------------

    @property
    def config(self) -> FrameworkConfig:
        with self._lock:
            return self._config

    @property
    def state(self) -> ExecutionState:
        with self._lock:
            return self._state

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._state == ExecutionState.READY

    @property
    def model_id(self) -> str:
        with self._lock:
            return self._active_model_id

    @property
    def active_executions(self) -> int:
        with self._lock:
            return self._active_executions

    @property
    def metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._metrics)

    @property
    def status(self) -> Dict[str, Any]:
        """Returns observable execution status and metrics from the status tracker."""
        with self._lock:
            if self.status_tracker and hasattr(self.status_tracker, "get_status"):
                s = dict(self.status_tracker.get_status())
                s["state"] = self._state.value
                return s
            return {
                "state": self._state.value,
                "metrics": dict(self._metrics),
                "is_ready": self.is_ready,
                "active_executions": self._active_executions,
            }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _notify_state_change(self, prev_state: ExecutionState, new_state: ExecutionState, reason: str = "") -> None:
        """Notifies the status tracker of a lifecycle state transition."""
        if self.status_tracker and hasattr(self.status_tracker, "record_state_change"):
            try:
                self.status_tracker.record_state_change(prev_state, new_state, reason=reason)
            except Exception as e:
                logger.warning(f"Status tracker record_state_change error: {e}")

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Returns or lazily instantiates the bounded thread pool executor for stage execution."""
        with self._lock:
            if self._executor is None:
                max_workers = max(4, (os.cpu_count() or 1) * 2)
                self._executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="ml_framework_worker"
                )
            return self._executor

    def _wait_for_in_flight(self, timeout: Optional[float] = 5.0) -> bool:
        """
        Waits for active in-flight executions to drain. Must be called while holding self._lock.
        Returns True if fully drained, False if timeout expired.
        """
        deadline = time.time() + timeout if timeout is not None else None
        while self._active_executions > 0:
            remaining = deadline - time.time() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                logger.warning(
                    f"In-flight executions did not drain within {timeout}s timeout "
                    f"({self._active_executions} remaining)."
                )
                return False
            self._drain_cond.wait(timeout=remaining)
        return True

    def _record_success_metrics(self, duration_ms: float) -> None:
        """Updates internal execution metrics on success under lock."""
        with self._lock:
            self._metrics["total_executions"] += 1
            self._metrics["successful_executions"] += 1
            self._metrics["last_duration_ms"] = duration_ms
            self._metrics["total_duration_ms"] += duration_ms

    def _record_failure_metrics(self, duration_ms: float) -> None:
        """Updates internal execution metrics on failure under lock."""
        with self._lock:
            self._metrics["total_executions"] += 1
            self._metrics["failed_executions"] += 1
            self._metrics["last_duration_ms"] = duration_ms
            self._metrics["total_duration_ms"] += duration_ms

    # -------------------------------------------------------------------------
    # Lifecycle Management
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Initializes the execution environment, validates configuration, and prepares components.
        Transitions state from UNINITIALIZED / FAILED -> INITIALIZING -> READY.
        Thread-safe and idempotent if already READY.
        """
        with self._lock:
            if self._state == ExecutionState.READY:
                logger.info("ExecutionManager is already initialized and READY.")
                return True
            if self._state == ExecutionState.INITIALIZING:
                raise StateError("Cannot initialize: ExecutionManager is already INITIALIZING.")
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot initialize: ExecutionManager is currently RELOADING.")
            if self._draining:
                raise StateError("Cannot initialize: ExecutionManager is currently draining.")
            if self._active_executions > 0:
                raise StateError("Cannot initialize while executions are in-flight.")

            prev_init_state = self._state
            self._state = ExecutionState.INITIALIZING

        self._notify_state_change(prev_init_state, ExecutionState.INITIALIZING, "initialize() started")

        try:
            # Stage A: Input Configuration Validation (Task 1 hook)
            input_src = (
                self._config.input_config.source
                if self._config.input_config and self._config.input_config.source is not None
                else self._config.input_source
            )
            if input_src is not None and self.input_processor:
                try:
                    self.input_processor.validate_source(input_src)
                except Exception as in_err:
                    self.shutdown()
                    raise in_err

            # Stage B: Environment validation (Task 3 hook)
            if self.env_manager:
                self.env_manager.validate_environment(self._config)

            # Stage C: Model loading (MinIO-to-Triton retrieval & staging)
            loaded_instance = None
            from .inference.triton_engine import TritonInferenceEngine
            is_triton = isinstance(self.inference_engine, TritonInferenceEngine)

            # Resolve model_source if model_bucket and model_key were provided
            model_src = self._config.model_source
            if not model_src and self._config.model_bucket and self._config.model_key:
                model_src = f"s3://{self._config.model_bucket.strip()}/{self._config.model_key.lstrip('/')}"
                self._config.model_source = model_src

            if self.model_loader and model_src:
                supported_schemes = getattr(self.model_loader, "SUPPORTED_STORAGE_SCHEMES", {"s3", "minio", "seaweedfs"})
                is_remote = any(str(model_src).startswith(f"{s}://") for s in supported_schemes)
                can_load_for_triton = callable(getattr(self.model_loader, "load_for_triton", None))
                is_nemo_or_triton = (
                    is_remote
                    or bool(self._config.model_bucket)
                    or self._config.framework in ("triton", "nemo")
                    or (isinstance(model_src, str) and model_src.lower().endswith(".nemo"))
                    or bool(self._config.triton_config.model_name and self._config.framework in ("triton", "nemo", "speech", "auto"))
                )
                use_triton_loading = is_triton and can_load_for_triton and is_nemo_or_triton
                if use_triton_loading:
                    loaded_instance = self.model_loader.load_for_triton(
                        model_src, self._config, self.inference_engine
                    )
                else:
                    loaded_instance = self.model_loader.load(
                        model_src, self._config
                    )

            # Stage D: Triton Engine readiness verification
            is_triton_served = (
                (isinstance(loaded_instance, dict) and loaded_instance.get("is_triton_served", False))
                or loaded_instance is None
            )
            if is_triton and is_triton_served:
                model_to_verify = (
                    self._config.triton_config.model_name
                    or (loaded_instance.get("model_name") if isinstance(loaded_instance, dict) else None)
                    or self._config.model_name
                )
                if self._config.triton_config.check_model_ready and model_to_verify:
                    self.inference_engine.verify_readiness(
                        model_name=model_to_verify,
                        model_version=self._config.triton_config.model_version
                    )

            with self._lock:
                self._active_model_instance = loaded_instance
                if isinstance(loaded_instance, dict) and loaded_instance.get("model_name"):
                    self._active_model_id = str(loaded_instance["model_name"])
                self._state = ExecutionState.READY

            self._notify_state_change(ExecutionState.INITIALIZING, ExecutionState.READY, "initialize() successful")
            logger.info(f"ExecutionManager initialized successfully (Model ID: {self._active_model_id}).")
            return True

        except Exception as e:
            with self._lock:
                self._state = ExecutionState.FAILED
            self._notify_state_change(ExecutionState.INITIALIZING, ExecutionState.FAILED, f"initialize() failed: {e}")
            logger.error(f"ExecutionManager initialization failed: {str(e)}", exc_info=True)
            if isinstance(e, MLFrameworkError):
                raise
            raise MLFrameworkError(f"Initialization failed: {str(e)}") from e

    def execute(
        self,
        input_data: Any,
        context: Optional[ExecutionContext] = None
    ) -> ExecutionResult:
        """
        Executes the end-to-end execution pipeline on input data:
        1. Context validation & correlation ID assignment
        2. Input Processing
        3. Inference Engine forward pass
        4. Output Processing
        5. Metrics & Status Telemetry

        Thread-safe: supports concurrent execution across threads.
        Enforces timeout_seconds honestly: returns ExecutionTimeoutError on timeout,
        while ensuring late worker completion never causes duplicate metric updates
        or duplicate callbacks.
        """
        start_time = time.time()
        ctx = context if context is not None else ExecutionContext()

        # Enforce lifecycle state and register in-flight task under lock
        with self._lock:
            if self._state == ExecutionState.UNINITIALIZED:
                raise StateError("Cannot execute: ExecutionManager is UNINITIALIZED. Call initialize() first.")
            if self._state == ExecutionState.STOPPED:
                raise StateError("Cannot execute: ExecutionManager is STOPPED.")
            if self._state == ExecutionState.INITIALIZING:
                raise StateError("Cannot execute: ExecutionManager is currently INITIALIZING.")
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot execute: ExecutionManager is currently RELOADING.")
            if self._draining:
                raise StateError("Cannot execute: ExecutionManager is draining in-flight tasks.")
            if self._state == ExecutionState.FAILED:
                raise StateError("Cannot execute: ExecutionManager is in FAILED state. Re-initialization required.")
            if self._state == ExecutionState.PAUSED:
                return ExecutionResult(
                    success=False,
                    output=None,
                    duration_ms=0.0,
                    context=ctx,
                    state=self._state,
                    error="ExecutionManager is currently PAUSED. Message consumption should wait.",
                    error_type="StateError",
                    applied_model_id=self._active_model_id,
                )
            if self._state != ExecutionState.READY:
                raise StateError(f"Cannot execute: ExecutionManager is in invalid state '{self._state.value}'.")

            # Increment in-flight execution counter and bind model to this execution
            self._active_executions += 1
            bound_model_instance = self._active_model_instance
            model_id_snapshot = self._active_model_id

        # Worker closure executing sequential pipeline stages
        def _run_pipeline() -> Any:
            try:
                # Stage 1: Input Processing (Task 2 hook)
                s1_start = time.time()
                preprocessed = self._process_input(input_data, ctx)
                ctx.record_stage("input_processing", (time.time() - s1_start) * 1000)

                # Stage 2: Inference Engine (Task 5 hook) - uses bound model instance
                s2_start = time.time()
                raw_output = self._run_inference(bound_model_instance, preprocessed, ctx)
                ctx.record_stage("inference", (time.time() - s2_start) * 1000)

                # Stage 3: Output Processing (Task 6 hook)
                s3_start = time.time()
                final_output = self._process_output(raw_output, ctx)
                ctx.record_stage("output_processing", (time.time() - s3_start) * 1000)

                return final_output
            finally:
                # Decrement in-flight count when worker thread actually finishes
                with self._lock:
                    self._active_executions -= 1
                    self._drain_cond.notify_all()

        timeout_sec = self._config.timeout_seconds
        executor = self._get_executor()

        try:
            future = executor.submit(_run_pipeline)
            final_output = future.result(timeout=timeout_sec)
            duration_ms = round((time.time() - start_time) * 1000, 3)

            # Record metrics strictly once on success
            self._record_success_metrics(duration_ms)

            result = ExecutionResult(
                success=True,
                output=final_output,
                duration_ms=duration_ms,
                context=ctx,
                state=ExecutionState.READY,
                applied_model_id=model_id_snapshot,
            )

            # Stage 4: Telemetry recording (Task 7 hook)
            if self.status_tracker:
                try:
                    self.status_tracker.record_success(result)
                except Exception as st_err:
                    logger.warning(f"Status tracker record_success error: {st_err}")
            self._notify_callbacks(result)

            return result

        except concurrent.futures.TimeoutError:
            duration_ms = round((time.time() - start_time) * 1000, 3)
            timeout_msg = f"Execution exceeded configured timeout of {timeout_sec}s"
            timeout_err = ExecutionTimeoutError(
                timeout_msg,
                timeout_seconds=timeout_sec,
                details={"timeout_seconds": timeout_sec, "correlation_id": ctx.correlation_id}
            )
            logger.error(f"Execution timed out for correlation_id={ctx.correlation_id}: {timeout_msg}")

            # Record failure metrics strictly once on timeout
            self._record_failure_metrics(duration_ms)

            # Evaluate auto_cleanup if configured
            if self._config.auto_cleanup and self.resource_manager:
                try:
                    self.resource_manager.cleanup()
                except Exception as clean_err:
                    logger.warning(f"Auto-cleanup failed after execution timeout: {clean_err}")

            result = ExecutionResult(
                success=False,
                output=None,
                duration_ms=duration_ms,
                context=ctx,
                state=ExecutionState.FAILED,
                error=timeout_msg,
                error_type="ExecutionTimeoutError",
                applied_model_id=model_id_snapshot,
            )

            if self.status_tracker:
                try:
                    self.status_tracker.record_failure(timeout_err, ctx)
                except Exception as st_err:
                    logger.warning(f"Status tracker record_failure error: {st_err}")
            self._notify_callbacks(result)

            return result

        except Exception as e:
            duration_ms = round((time.time() - start_time) * 1000, 3)
            logger.error(f"Execution failed for correlation_id={ctx.correlation_id}: {str(e)}", exc_info=True)

            # Record failure metrics strictly once on error
            self._record_failure_metrics(duration_ms)

            # Evaluate auto_cleanup if configured
            if self._config.auto_cleanup and self.resource_manager:
                try:
                    self.resource_manager.cleanup()
                except Exception as clean_err:
                    logger.warning(f"Auto-cleanup failed after execution error: {clean_err}")

            result = ExecutionResult(
                success=False,
                output=None,
                duration_ms=duration_ms,
                context=ctx,
                state=ExecutionState.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                applied_model_id=model_id_snapshot,
            )

            if self.status_tracker:
                try:
                    self.status_tracker.record_failure(e, ctx)
                except Exception as st_err:
                    logger.warning(f"Status tracker record_failure error: {st_err}")
            self._notify_callbacks(result)

            return result

    # -------------------------------------------------------------------------
    # Pipeline Stage Hooks (Default Passthroughs / Placeholders)
    # -------------------------------------------------------------------------

    def _process_input(self, input_data: Any, context: ExecutionContext) -> Any:
        """Invokes the injected InputProcessor or returns data directly."""
        if self.input_processor:
            return self.input_processor.process(input_data, context)
        return input_data

    def _run_inference(
        self,
        model_instance: Any,
        preprocessed_input: Any,
        context: ExecutionContext
    ) -> Any:
        """
        Invokes the injected InferenceEngine or bound model runner under an inference lock.

        Concurrency Policy:
        Shared model instances and inference engines are not guaranteed to be re-entrant.
        Inference is serialized via self._inference_lock to prevent simultaneous execution
        on the same model instance, while permitting input/output stages to run concurrently.
        The lifecycle lock (self._lock) is NOT held during inference.
        """
        with self._inference_lock:
            if self.inference_engine:
                return self.inference_engine.infer(
                    model_instance, preprocessed_input, context
                )
            if model_instance and hasattr(model_instance, "predict"):
                raise InferenceError(
                    "Direct local execution via model adapter is disabled. "
                    "Production inference must be delegated to NVIDIA Triton Inference Server.",
                    details={"correlation_id": context.correlation_id},
                )
            return preprocessed_input

    def _process_output(self, raw_output: Any, context: ExecutionContext) -> Any:
        """Invokes the injected OutputProcessor or returns raw output directly."""
        if self.output_processor:
            return self.output_processor.process(raw_output, context)
        return raw_output

    # -------------------------------------------------------------------------
    # Operational Controls
    # -------------------------------------------------------------------------

    def pause(self) -> None:
        """Sets state to PAUSED. Signals consumers to hold off message delivery."""
        with self._lock:
            if self._state == ExecutionState.PAUSED:
                return
            if self._state == ExecutionState.STOPPED:
                raise StateError("Cannot pause: ExecutionManager is STOPPED.")
            if self._state == ExecutionState.UNINITIALIZED:
                raise StateError("Cannot pause: ExecutionManager is UNINITIALIZED.")
            if self._state == ExecutionState.FAILED:
                raise StateError("Cannot pause: ExecutionManager is in FAILED state.")
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot pause: ExecutionManager is currently RELOADING.")
            if self._draining:
                raise StateError("Cannot pause: ExecutionManager is currently draining.")
            prev_state = self._state
            self._state = ExecutionState.PAUSED

        self._notify_state_change(prev_state, ExecutionState.PAUSED, "pause() called")
        logger.info("ExecutionManager is now PAUSED.")

    def resume(self) -> None:
        """Restores state to READY from PAUSED."""
        with self._lock:
            if self._state != ExecutionState.PAUSED:
                raise StateError(f"Cannot resume: ExecutionManager is in '{self._state.value}' state, expected PAUSED.")
            prev_state = self._state
            self._state = ExecutionState.READY

        self._notify_state_change(prev_state, ExecutionState.READY, "resume() called")
        logger.info("ExecutionManager is now resumed and READY.")

    def reload(
        self,
        new_source: Optional[str] = None,
        timeout: Optional[float] = None,
        model_bucket: Optional[str] = None,
        model_key: Optional[str] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        input_type: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Safely reloads or switches active model with rollback preservation on failure.
        1. Validates current state and target source.
        2. Sets transitional state RELOADING to block new executions from entering.
        3. Preserves active model and config as backup snapshot.
        4. Loads new model instance into a staging reference outside the lock.
        5. Waits for in-flight executions on the old model to drain.
        6. Atomically commits the new model.
        7. Unloads old model safely after drain succeeds.
        8. On failure, restores previous model and state without service loss.
        """
        with self._lock:
            if self._state == ExecutionState.STOPPED:
                raise StateError("Cannot reload: ExecutionManager is STOPPED.")
            if self._state == ExecutionState.INITIALIZING:
                raise StateError("Cannot reload: ExecutionManager is currently INITIALIZING.")
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot reload: ExecutionManager is already RELOADING.")
            if self._draining:
                raise StateError("Cannot reload: ExecutionManager is currently draining.")
            if self._state == ExecutionState.UNINITIALIZED:
                raise StateError("Cannot reload: ExecutionManager is UNINITIALIZED. Call initialize() first.")

            bkt = model_bucket or self._config.model_bucket
            k = model_key or (kwargs.get("model_dir") if "model_dir" in kwargs else self._config.model_key)
            if new_source:
                target_source = new_source
            elif bkt and k:
                target_source = f"s3://{bkt.strip()}/{k.lstrip('/')}"
            else:
                target_source = self._config.model_source

            if not target_source:
                raise ConfigurationError("Cannot reload: no model_source specified in config or arguments.")

            prev_state = self._state
            prev_source = self._config.model_source
            prev_instance = self._active_model_instance
            prev_id = self._active_model_id

            # Set transitional state to block new executions from entering
            self._state = ExecutionState.RELOADING

        self._notify_state_change(prev_state, ExecutionState.RELOADING, "reload() started")

        # Stage new model outside the lock so active executions are not blocked during model download/loading
        staged_instance = None
        if self.model_loader and target_source:
            try:
                temp_cfg = copy.deepcopy(self._config)
                temp_cfg.model_source = target_source
                if model_bucket:
                    temp_cfg.model_bucket = model_bucket
                if model_key:
                    temp_cfg.model_key = model_key
                if model_name:
                    temp_cfg.model_name = model_name
                    temp_cfg.triton_config.model_name = model_name
                if task:
                    temp_cfg.task = task
                if input_type:
                    temp_cfg.input_type = input_type
                for extra_k, extra_v in kwargs.items():
                    if hasattr(temp_cfg, extra_k):
                        setattr(temp_cfg, extra_k, extra_v)

                from .inference.triton_engine import TritonInferenceEngine
                is_triton = isinstance(self.inference_engine, TritonInferenceEngine)
                can_load_for_triton = callable(getattr(self.model_loader, "load_for_triton", None))
                is_nemo_or_triton = (
                    any(str(target_source).startswith(f"{s}://") for s in getattr(self.model_loader, "SUPPORTED_STORAGE_SCHEMES", {"s3", "minio", "seaweedfs"}))
                    or bool(temp_cfg.model_bucket)
                    or temp_cfg.framework in ("triton", "nemo")
                    or (isinstance(target_source, str) and target_source.lower().endswith(".nemo"))
                    or bool(temp_cfg.triton_config.model_name and temp_cfg.framework in ("triton", "nemo", "speech", "auto"))
                )
                use_triton_loading = is_triton and can_load_for_triton and is_nemo_or_triton
                if use_triton_loading:
                    staged_instance = self.model_loader.load_for_triton(target_source, temp_cfg, self.inference_engine)
                else:
                    staged_instance = self.model_loader.load(target_source, temp_cfg)

            except Exception as load_err:
                logger.error(f"Model reload failed to load candidate model from '{target_source}': {load_err}")
                with self._lock:
                    self._state = prev_state
                    self._config.model_source = prev_source
                    self._active_model_instance = prev_instance
                    self._active_model_id = prev_id
                self._notify_state_change(ExecutionState.RELOADING, prev_state, f"reload failed: {load_err}")
                if isinstance(load_err, MLFrameworkError):
                    raise
                raise ModelLoadError(f"Failed to reload model from '{target_source}': {load_err}") from load_err

        # Drain in-flight executions and commit swap under lock
        with self._lock:
            drain_timeout = timeout if timeout is not None else self._config.timeout_seconds
            self._draining = True
            try:
                drained = self._wait_for_in_flight(timeout=drain_timeout)
            finally:
                self._draining = False

            if not drained:
                logger.error(
                    f"Reload aborted: in-flight executions failed to drain within {drain_timeout}s timeout."
                )
                # Rollback state and retain previous model
                self._config.model_source = prev_source
                self._active_model_instance = prev_instance
                self._active_model_id = prev_id
                self._state = prev_state
                self._notify_state_change(ExecutionState.RELOADING, prev_state, "reload drain timeout rollback")
                # Unload staged candidate instance to avoid resource leak
                if staged_instance is not None and self.model_loader:
                    try:
                        self.model_loader.unload(staged_instance)
                    except Exception as staging_clean_err:
                        logger.warning(f"Error unloading candidate model during rollback: {staging_clean_err}")
                raise ModelLoadError(
                    f"Cannot reload: in-flight executions failed to drain within {drain_timeout}s timeout"
                )

            # Atomically commit the new model
            self._config.model_source = target_source
            if model_bucket:
                self._config.model_bucket = model_bucket
            if model_key:
                self._config.model_key = model_key
            if model_name:
                self._config.model_name = model_name
                self._config.triton_config.model_name = model_name
            if task:
                self._config.task = task
            if input_type:
                self._config.input_type = input_type
            self._active_model_instance = staged_instance
            if isinstance(staged_instance, dict) and staged_instance.get("model_name"):
                self._active_model_id = str(staged_instance["model_name"])
            else:
                self._active_model_id = (
                    f"{self._config.model_name}:{self._config.model_version}"
                    if self._config.model_name and self._config.model_version
                    else target_source
                )
            self._state = ExecutionState.READY
        self._notify_state_change(ExecutionState.RELOADING, ExecutionState.READY, "reload committed")
        logger.info(f"Model reload committed successfully (New Model ID: {self._active_model_id}).")

        # Safely unload previous model and clean up resources outside lock (guaranteed 0 in-flight executions on prev_instance)
        if prev_instance is not None and self.model_loader:
            try:
                self.model_loader.unload(prev_instance)
            except Exception as old_unload_err:
                logger.warning(f"Error unloading previous model after reload: {old_unload_err}")
        if self.resource_manager:
            try:
                self.resource_manager.cleanup()
            except Exception as clean_err:
                logger.warning(f"Error running resource cleanup after reload: {clean_err}")

        return True

    def process(
        self,
        input_data: Any,
        context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Public model processing entry point conforming to:
            model = ml.load(...)
            result = model.process(input_data)

        Delegates to execute() and returns structured ExecutionResult.
        """
        return self.execute(input_data, context=context)

    def unload(self, timeout: Optional[float] = None) -> None:
        """
        Unloads active model instances from memory and accelerators.
        Waits for in-flight executions to drain and sets state to UNINITIALIZED
        without terminating the manager process.
        """
        with self._lock:
            if self._state == ExecutionState.STOPPED:
                logger.info("ExecutionManager is STOPPED; unload is a no-op.")
                return
            if self._state == ExecutionState.UNINITIALIZED and self._active_model_instance is None:
                logger.info("ExecutionManager is already UNINITIALIZED.")
                return
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot unload: ExecutionManager is currently RELOADING.")
            if self._draining:
                raise StateError("Cannot unload: ExecutionManager is currently draining.")

            logger.info(f"Unloading active model: {self._active_model_id}")
            prev_state = self._state
            drain_timeout = timeout if timeout is not None else self._config.timeout_seconds
            self._draining = True
            try:
                drained = self._wait_for_in_flight(timeout=drain_timeout)
            finally:
                self._draining = False

            if not drained:
                logger.error(
                    f"Model unload aborted: in-flight executions failed to drain within {drain_timeout}s timeout."
                )
                self._state = prev_state
                raise ResourceCleanupError(
                    f"Cannot unload model: in-flight executions failed to drain within {drain_timeout}s timeout"
                )

            active_model = self._active_model_instance
            self._active_model_instance = None
            self._state = ExecutionState.UNINITIALIZED

        self._notify_state_change(prev_state, ExecutionState.UNINITIALIZED, "unload() completed")

        # Run independent cleanup hooks outside the lock
        if self.model_loader and active_model is not None:
            try:
                self.model_loader.unload(active_model)
            except Exception as e:
                logger.warning(f"Error during model loader unload: {e}", exc_info=True)

        if self.resource_manager:
            try:
                self.resource_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error during resource manager cleanup: {e}", exc_info=True)

        logger.info("Active model unloaded; state is now UNINITIALIZED.")

    def stop(self) -> None:
        """
        Lifecycle stop operation:
            model.stop()

        Completely terminates the manager and releases all allocated resources.
        """
        self.shutdown()

    @classmethod
    def load(
        cls,
        model_source: Optional[str] = None,
        config: Optional[Union[FrameworkConfig, Dict[str, Any]]] = None,
        **kwargs
    ) -> "ExecutionManager":
        """Convenience factory method instantiating and initializing an ExecutionManager."""
        from .ml import load as ml_load
        return ml_load(model_source=model_source, config=config, **kwargs)

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """
        Shuts down the manager, drains active executions, unloads models,
        triggers resource cleanup, and terminates the thread pool executor.
        Sets state to STOPPED. Idempotent.
        """
        with self._lock:
            if self._state == ExecutionState.STOPPED:
                logger.info("ExecutionManager is already STOPPED.")
                return
            if self._state == ExecutionState.RELOADING:
                raise StateError("Cannot shutdown: ExecutionManager is currently RELOADING.")
            if self._draining:
                raise StateError("Cannot shutdown: ExecutionManager is currently draining.")

            logger.info("Shutting down ExecutionManager...")
            prev_state = self._state
            drain_timeout = timeout if timeout is not None else self._config.timeout_seconds
            self._draining = True
            try:
                drained = self._wait_for_in_flight(timeout=drain_timeout)
            finally:
                self._draining = False

            if not drained:
                logger.error(
                    f"Shutdown aborted: in-flight executions failed to drain within {drain_timeout}s timeout."
                )
                self._state = prev_state
                raise ResourceCleanupError(
                    f"Cannot shutdown: in-flight executions failed to drain within {drain_timeout}s timeout"
                )

            active_model = self._active_model_instance
            self._active_model_instance = None
            self._state = ExecutionState.STOPPED
            executor = self._executor
            self._executor = None

        self._notify_state_change(prev_state, ExecutionState.STOPPED, "shutdown() completed")

        # Terminate thread pool executor
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception as e:
                logger.warning(f"Error shutting down internal executor: {e}")

        # Run independent cleanup hooks
        if self.model_loader and active_model is not None:
            try:
                self.model_loader.unload(active_model)
            except Exception as e:
                logger.warning(f"Error during model loader unload at shutdown: {e}", exc_info=True)

        if self.model_loader and hasattr(self.model_loader, "close"):
            try:
                self.model_loader.close()
            except Exception as e:
                logger.warning(f"Error during model loader close at shutdown: {e}", exc_info=True)

        if self.resource_manager:
            try:
                self.resource_manager.cleanup()
            except Exception as e:
                logger.warning(f"Error during resource manager cleanup at shutdown: {e}", exc_info=True)

        if self.input_processor and hasattr(self.input_processor, "close"):
            try:
                self.input_processor.close()
            except Exception as e:
                logger.warning(f"Error during input processor cleanup at shutdown: {e}", exc_info=True)

        logger.info("ExecutionManager is now STOPPED.")

    def register_status_callback(self, callback: Callable[[ExecutionResult], None]) -> None:
        """Registers a callback hook invoked on every execution completion."""
        if not callable(callback):
            raise ValueError(f"Expected callable callback, got {type(callback).__name__}")
        with self._lock:
            self._status_callbacks.append(callback)

    def _notify_callbacks(self, result: ExecutionResult) -> None:
        """Notifies registered callbacks defensively taking a snapshot of the callback list."""
        with self._lock:
            callbacks = list(self._status_callbacks)
        for cb in callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.warning(f"Status callback {cb} error: {e}", exc_info=True)
