"""
Unit tests for ExecutionManager lifecycle, orchestration, injection, and error handling.
"""

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, List

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import time
import threading
from ml_framework import (
    ExecutionManager,
    FrameworkConfig,
    ExecutionContext,
    ExecutionState,
    ExecutionResult,
)
from ml_framework.errors import (
    ConfigurationError,
    StateError,
    InferenceError,
    ExecutionTimeoutError,
    ModelLoadError,
    MLFrameworkError,
    ResourceCleanupError,
)
from ml_framework.interfaces import (
    BaseEnvironmentManager,
    BaseModelLoader,
    BaseInputProcessor,
    BaseInferenceEngine,
    BaseOutputProcessor,
    BaseStatusTracker,
    BaseResourceManager,
)


# -----------------------------------------------------------------------------
# Mock Implementations for Dependency Injection Verification
# -----------------------------------------------------------------------------

class MockEnvManager(BaseEnvironmentManager):
    def __init__(self):
        self.validated = False

    def validate_environment(self, config: FrameworkConfig) -> bool:
        self.validated = True
        return True

    def get_device(self) -> str:
        return "cpu"


class MockModelLoader(BaseModelLoader):
    def __init__(self, fail_on: Optional[str] = None, should_fail_unload: bool = False):
        self.loaded = False
        self.unloaded = False
        self.fail_on = fail_on
        self.should_fail_unload = should_fail_unload
        self.loaded_sources = []
        self.unloaded_instances = []
        self.closed = False

    def load(self, model_source: str, config: Optional[FrameworkConfig] = None) -> Any:
        if self.fail_on and model_source == self.fail_on:
            raise ModelLoadError(f"Simulated load failure for source: {model_source}")
        self.loaded = True
        self.loaded_sources.append(model_source)
        return {"loaded_model_source": model_source}

    def unload(self, model_instance: Any) -> None:
        self.unloaded = True
        self.unloaded_instances.append(model_instance)
        if self.should_fail_unload:
            raise RuntimeError("Simulated unload error in model loader")

    def close(self) -> None:
        self.closed = True


class MockInputProcessor(BaseInputProcessor):
    def process(self, input_data: Any, context: ExecutionContext) -> Any:
        if isinstance(input_data, str):
            return input_data.strip().upper()
        return input_data


class MockInferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        should_fail: bool = False,
        delay: float = 0.0,
        sync_event: Optional[threading.Event] = None
    ):
        self.should_fail = should_fail
        self.delay = delay
        self.sync_event = sync_event
        self.call_count = 0

    def infer(self, model_instance: Any, preprocessed_input: Any, context: ExecutionContext) -> Any:
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        if self.sync_event is not None:
            self.sync_event.wait(timeout=2.0)
        if self.should_fail:
            raise InferenceError("Synthetic inference failure for testing")
        return f"INFERRED_{preprocessed_input}"


class MockOutputProcessor(BaseOutputProcessor):
    def process(self, raw_output: Any, context: ExecutionContext) -> Any:
        return {"formatted_data": raw_output, "processed": True}


class MockStatusTracker(BaseStatusTracker):
    def __init__(self):
        self.successes = []
        self.failures = []

    def record_success(self, result: ExecutionResult) -> None:
        self.successes.append(result)

    def record_failure(self, error: Exception, context: ExecutionContext) -> None:
        self.failures.append((error, context))


class MockResourceManager(BaseResourceManager):
    def __init__(self, should_fail: bool = False):
        self.cleaned_up = False
        self.cleanup_count = 0
        self.should_fail = should_fail

    def cleanup(self) -> None:
        self.cleaned_up = True
        self.cleanup_count += 1
        if self.should_fail:
            raise RuntimeError("Synthetic resource cleanup failure")


# -----------------------------------------------------------------------------
# Test Suite
# -----------------------------------------------------------------------------

class TestExecutionManager(unittest.TestCase):
    """Test suite for ExecutionManager core orchestration logic."""

    def test_default_instantiation(self):
        """Test instantiation with defaults."""
        manager = ExecutionManager()
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
        self.assertFalse(manager.is_ready)
        self.assertEqual(manager.metrics["total_executions"], 0)

    def test_invalid_config_rejection(self):
        """Test passing invalid config arguments."""
        with self.assertRaises(ConfigurationError):
            ExecutionManager(config=12345)

        with self.assertRaises(ConfigurationError):
            ExecutionManager(config={"batch_size": -1})

    def test_initialization_transitions(self):
        """Test that initialize() transitions state to READY."""
        manager = ExecutionManager()
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)

        success = manager.initialize()
        self.assertTrue(success)
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertTrue(manager.is_ready)

        # Idempotent call returns True
        self.assertTrue(manager.initialize())

    def test_execute_uninitialized_raises_state_error(self):
        """Test executing before initialize raises StateError."""
        manager = ExecutionManager()
        with self.assertRaises(StateError) as cm:
            manager.execute("test input")
        self.assertIn("Call initialize() first", str(cm.exception))

    def test_execute_ready_default_passthrough(self):
        """Test executing without injected components acts as passthrough."""
        manager = ExecutionManager()
        manager.initialize()

        result = manager.execute("sample text")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "sample text")
        self.assertGreater(result.duration_ms, 0)
        self.assertEqual(result.applied_model_id, "unspecified")
        self.assertIn("input_processing", result.context.stage_durations)
        self.assertIn("inference", result.context.stage_durations)
        self.assertIn("output_processing", result.context.stage_durations)

        # Check metrics updated
        self.assertEqual(manager.metrics["total_executions"], 1)
        self.assertEqual(manager.metrics["successful_executions"], 1)
        self.assertEqual(manager.metrics["failed_executions"], 0)

    def test_pause_and_resume_lifecycle(self):
        """Test pause holds execution, resume restores execution."""
        manager = ExecutionManager()
        manager.initialize()

        manager.pause()
        self.assertEqual(manager.state, ExecutionState.PAUSED)
        self.assertFalse(manager.is_ready)

        # Executing while paused should return non-crash failure result
        result = manager.execute("data while paused")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "StateError")
        self.assertIn("PAUSED", result.error)

        # Resume
        manager.resume()
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertTrue(manager.is_ready)

        resumed_result = manager.execute("data after resume")
        self.assertTrue(resumed_result.success)
        self.assertEqual(resumed_result.output, "data after resume")

    def test_shutdown_lifecycle(self):
        """Test shutdown cleans up resources and marks manager STOPPED."""
        resource_mgr = MockResourceManager()
        model_loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://dummy_path")

        manager = ExecutionManager(
            config=cfg,
            model_loader=model_loader,
            resource_manager=resource_mgr
        )
        manager.initialize()
        self.assertTrue(model_loader.loaded)

        manager.shutdown()
        self.assertEqual(manager.state, ExecutionState.STOPPED)
        self.assertTrue(resource_mgr.cleaned_up)
        self.assertTrue(model_loader.unloaded)

        # Execution after stop should raise StateError
        with self.assertRaises(StateError):
            manager.execute("data after stop")

    def test_full_component_injection_pipeline(self):
        """Test injecting all components (Tasks 2-8 contracts)."""
        env_mgr = MockEnvManager()
        loader = MockModelLoader()
        in_proc = MockInputProcessor()
        infer_eng = MockInferenceEngine()
        out_proc = MockOutputProcessor()
        tracker = MockStatusTracker()
        res_mgr = MockResourceManager()

        cfg = FrameworkConfig(
            model_name="ner_roberta",
            model_version="1.0",
            model_source="minio://bucket/model.bin",
        )

        manager = ExecutionManager(
            config=cfg,
            env_manager=env_mgr,
            model_loader=loader,
            input_processor=in_proc,
            inference_engine=infer_eng,
            output_processor=out_proc,
            status_tracker=tracker,
            resource_manager=res_mgr,
        )

        # 1. Initialize
        manager.initialize()
        self.assertTrue(env_mgr.validated)
        self.assertTrue(loader.loaded)
        self.assertEqual(manager.model_id, "ner_roberta:1.0")

        # 2. Execute
        ctx = ExecutionContext(correlation_id="tx-9999")
        result = manager.execute("  raw entity text  ", context=ctx)

        self.assertTrue(result.success)
        # Expected pipeline transforms:
        # Input: "  raw entity text  " -> InputProcessor: "RAW ENTITY TEXT"
        # -> InferenceEngine: "INFERRED_RAW ENTITY TEXT"
        # -> OutputProcessor: {"formatted_data": "INFERRED_RAW ENTITY TEXT", "processed": True}
        self.assertEqual(
            result.output,
            {"formatted_data": "INFERRED_RAW ENTITY TEXT", "processed": True}
        )
        self.assertEqual(result.context.correlation_id, "tx-9999")
        self.assertEqual(len(tracker.successes), 1)
        self.assertEqual(tracker.successes[0].context.correlation_id, "tx-9999")

    def test_graceful_error_handling_and_status_tracking(self):
        """Test that unhandled stage exceptions return failure result without host crash."""
        failing_engine = MockInferenceEngine(should_fail=True)
        tracker = MockStatusTracker()

        manager = ExecutionManager(
            inference_engine=failing_engine,
            status_tracker=tracker,
        )
        manager.initialize()

        result = manager.execute("test data")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "InferenceError")
        self.assertIn("Synthetic inference failure", result.error)
        self.assertIsNone(result.output)

        # Tracker received failure
        self.assertEqual(len(tracker.failures), 1)
        err, ctx = tracker.failures[0]
        self.assertIsInstance(err, InferenceError)

        # Metrics recorded failure
        self.assertEqual(manager.metrics["failed_executions"], 1)

    def test_status_callback_invocation(self):
        """Test that registered status callbacks are notified."""
        manager = ExecutionManager()
        manager.initialize()

        received_results = []
        manager.register_status_callback(lambda res: received_results.append(res))

        manager.execute("item 1")
        manager.execute("item 2")

        self.assertEqual(len(received_results), 2)
        self.assertEqual(received_results[0].output, "item 1")
        self.assertEqual(received_results[1].output, "item 2")

    def test_invalid_lifecycle_operations(self):
        """Test strict state validation rejects invalid transitions."""
        manager = ExecutionManager()

        # 1. pause/resume before initialize raises StateError
        with self.assertRaises(StateError):
            manager.pause()
        with self.assertRaises(StateError):
            manager.resume()

        # 2. reload before initialize raises StateError
        with self.assertRaises(StateError):
            manager.reload("model://new")

        # 3. initialize -> READY
        manager.initialize()

        # 4. resume while READY raises StateError
        with self.assertRaises(StateError):
            manager.resume()

        # 5. shutdown -> STOPPED
        manager.shutdown()
        with self.assertRaises(StateError):
            manager.execute("cannot run when stopped")
        with self.assertRaises(StateError):
            manager.pause()
        with self.assertRaises(StateError):
            manager.reload("model://after_stop")

    def test_concurrent_executions_thread_safety(self):
        """Test concurrent execute() calls maintain exact metric consistency."""
        manager = ExecutionManager()
        manager.initialize()

        num_threads = 20
        results = [None] * num_threads

        def worker(idx):
            results[idx] = manager.execute(f"payload_{idx}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for i, res in enumerate(results):
            self.assertIsNotNone(res)
            self.assertTrue(res.success)
            self.assertEqual(res.output, f"payload_{i}")

        metrics = manager.metrics
        self.assertEqual(metrics["total_executions"], num_threads)
        self.assertEqual(metrics["successful_executions"], num_threads)
        self.assertEqual(metrics["failed_executions"], 0)

    def test_lifecycle_coordination_with_in_flight_executions(self):
        """Test unload() waits for in-flight executions to drain before releasing model."""
        sync_event = threading.Event()
        engine = MockInferenceEngine(sync_event=sync_event)
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://model_v1")

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        execution_finished = threading.Event()
        execution_result_container = []

        def slow_execute():
            res = manager.execute("slow payload")
            execution_result_container.append(res)
            execution_finished.set()

        t_exec = threading.Thread(target=slow_execute)
        t_exec.start()

        # Wait until the execution is actively in-flight
        for _ in range(50):
            if manager.active_executions > 0:
                break
            time.sleep(0.01)
        self.assertGreater(manager.active_executions, 0)

        # Call unload in a separate thread; it should wait for in-flight to complete
        unload_finished = threading.Event()

        def do_unload():
            manager.unload()
            unload_finished.set()

        t_unload = threading.Thread(target=do_unload)
        t_unload.start()

        # Verify unload has NOT completed yet because execution is held by sync_event
        time.sleep(0.05)
        self.assertFalse(unload_finished.is_set())
        self.assertFalse(loader.unloaded)

        # Release the in-flight execution
        sync_event.set()
        t_exec.join()
        t_unload.join()

        self.assertTrue(execution_finished.is_set())
        self.assertTrue(unload_finished.is_set())
        self.assertTrue(loader.unloaded)
        self.assertTrue(execution_result_container[0].success)
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)

    def test_execution_timeout_enforcement(self):
        """Test that stage execution exceeding timeout_seconds returns ExecutionTimeoutError."""
        engine = MockInferenceEngine(delay=0.3)
        cfg = FrameworkConfig(timeout_seconds=0.08)

        manager = ExecutionManager(config=cfg, inference_engine=engine)
        manager.initialize()

        ctx = ExecutionContext(correlation_id="corr-timeout-1")
        result = manager.execute("input text", context=ctx)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ExecutionTimeoutError")
        self.assertIn("Execution exceeded configured timeout", result.error)
        self.assertEqual(result.context.correlation_id, "corr-timeout-1")
        self.assertEqual(result.state, ExecutionState.FAILED)

        metrics = manager.metrics
        self.assertEqual(metrics["total_executions"], 1)
        self.assertEqual(metrics["failed_executions"], 1)
        self.assertEqual(metrics["successful_executions"], 0)

        # Clean shutdown (waits for background thread to exit cleanly)
        time.sleep(0.25)
        manager.shutdown()

    def test_no_duplicate_metrics_or_callbacks_after_timeout(self):
        """Test that late worker completion does not produce duplicate metrics or callbacks."""
        engine = MockInferenceEngine(delay=0.2)
        cfg = FrameworkConfig(timeout_seconds=0.05)

        manager = ExecutionManager(config=cfg, inference_engine=engine)
        manager.initialize()

        callbacks_received = []
        manager.register_status_callback(lambda res: callbacks_received.append(res))

        result = manager.execute("timed out payload")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ExecutionTimeoutError")

        # Verify exactly one callback received so far
        self.assertEqual(len(callbacks_received), 1)
        self.assertEqual(manager.metrics["total_executions"], 1)

        # Wait for delayed worker to completely exit
        time.sleep(0.3)

        # Assert no duplicate callbacks or metric increments occurred
        self.assertEqual(len(callbacks_received), 1)
        self.assertEqual(manager.metrics["total_executions"], 1)
        self.assertEqual(manager.metrics["successful_executions"], 0)
        self.assertEqual(manager.metrics["failed_executions"], 1)

        manager.shutdown()

    def test_reload_success_and_model_switching(self):
        """Test successful reload swaps model, unloads old model, and updates state."""
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://models/v1.tar.gz")

        manager = ExecutionManager(config=cfg, model_loader=loader)
        manager.initialize()

        self.assertEqual(manager.model_id, "local://models/v1.tar.gz")
        self.assertEqual(len(loader.loaded_sources), 1)
        self.assertEqual(len(loader.unloaded_instances), 0)

        # Reload with new source
        success = manager.reload("local://models/v2.tar.gz")
        self.assertTrue(success)
        self.assertEqual(manager.model_id, "local://models/v2.tar.gz")
        self.assertEqual(len(loader.loaded_sources), 2)
        self.assertEqual(len(loader.unloaded_instances), 1)
        self.assertEqual(loader.unloaded_instances[0], {"loaded_model_source": "local://models/v1.tar.gz"})
        self.assertEqual(manager.state, ExecutionState.READY)

    def test_reload_failure_preserves_previous_model(self):
        """Test failed reload rolls back and preserves the previously working model."""
        loader = MockModelLoader(fail_on="bad://broken_source")
        cfg = FrameworkConfig(model_source="good://working_source")

        manager = ExecutionManager(config=cfg, model_loader=loader)
        manager.initialize()

        self.assertEqual(manager.model_id, "good://working_source")

        # Attempt reload with broken source
        with self.assertRaises(ModelLoadError):
            manager.reload("bad://broken_source")

        # Verify previous model is preserved and manager remains operational
        self.assertEqual(manager.model_id, "good://working_source")
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(len(loader.unloaded_instances), 0)

        # Verify execution still works with the preserved model
        res = manager.execute("test after failed reload")
        self.assertTrue(res.success)
        self.assertEqual(res.output, "test after failed reload")

    def test_cleanup_hook_isolation_on_exception(self):
        """Test that failure in model_loader.unload does not skip resource_manager.cleanup."""
        loader = MockModelLoader(should_fail_unload=True)
        res_mgr = MockResourceManager()
        cfg = FrameworkConfig(model_source="local://dummy")

        manager = ExecutionManager(config=cfg, model_loader=loader, resource_manager=res_mgr)
        manager.initialize()

        # Unload should catch loader exception and proceed with resource cleanup
        manager.unload()
        self.assertTrue(res_mgr.cleaned_up)
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)

        # Shutdown should also proceed with resource cleanup
        res_mgr.cleaned_up = False
        manager.shutdown()
        self.assertTrue(res_mgr.cleaned_up)
        self.assertEqual(manager.state, ExecutionState.STOPPED)

    def test_auto_cleanup_triggered_on_execution_failure(self):
        """Test that auto_cleanup triggers resource cleanup when stage fails."""
        failing_engine = MockInferenceEngine(should_fail=True)
        res_mgr = MockResourceManager()
        cfg = FrameworkConfig(auto_cleanup=True)

        manager = ExecutionManager(config=cfg, inference_engine=failing_engine, resource_manager=res_mgr)
        manager.initialize()

        result = manager.execute("fail input")
        self.assertFalse(result.success)
        self.assertTrue(res_mgr.cleaned_up)
        self.assertGreaterEqual(res_mgr.cleanup_count, 1)

    def test_callback_isolation_on_error(self):
        """Test that an exception in one callback does not prevent subsequent callbacks from running."""
        manager = ExecutionManager()
        manager.initialize()

        def buggy_callback(res):
            raise RuntimeError("Callback bug!")

        received = []
        def healthy_callback(res):
            received.append(res)

        manager.register_status_callback(buggy_callback)
        manager.register_status_callback(healthy_callback)

        res = manager.execute("data for callback")
        self.assertTrue(res.success)
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].output, "data for callback")

    def test_state_restoration_after_initialization_failure(self):
        """Test that initialization failure sets FAILED state and can be recovered."""
        cfg = FrameworkConfig(input_source="/non/existent/critical/file.txt")
        manager = ExecutionManager(config=cfg)

        with self.assertRaises(MLFrameworkError):
            manager.initialize()

        self.assertEqual(manager.state, ExecutionState.FAILED)

        # Reconfigure with valid source and reinitialize
        cfg.input_source = None
        cfg.input_config.source = None
        manager.initialize()
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertTrue(manager.is_ready)

    def test_s1_reload_drain_timeout_aborts_and_preserves_old_model(self):
        """Test reload drain timeout aborts swap, preserves old model, and unloads staged model."""
        sync_event = threading.Event()
        engine = MockInferenceEngine(sync_event=sync_event)
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://v1_source", timeout_seconds=0.05)

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        # Start an in-flight execution that will block on sync_event
        t_exec = threading.Thread(target=lambda: manager.execute("in-flight payload"))
        t_exec.start()

        # Wait until execution is actively registered
        for _ in range(50):
            if manager.active_executions > 0:
                break
            time.sleep(0.01)
        self.assertEqual(manager.active_executions, 1)

        # Attempt reload with short drain timeout
        with self.assertRaises(ModelLoadError) as ctx:
            manager.reload("local://v2_candidate", timeout=0.05)

        self.assertIn("in-flight executions failed to drain", str(ctx.exception))

        # Assert old model is preserved and manager remains in READY state
        self.assertEqual(manager.model_id, "local://v1_source")
        self.assertEqual(manager.state, ExecutionState.READY)

        # Verify old model was NOT unloaded
        unloaded_sources = [m.get("loaded_model_source") for m in loader.unloaded_instances]
        self.assertNotIn("local://v1_source", unloaded_sources)

        # Verify staged candidate was cleaned up to avoid resource leaks
        self.assertIn("local://v2_candidate", unloaded_sources)

        # Release in-flight execution and clean up
        sync_event.set()
        t_exec.join()
        manager.shutdown()

    def test_s1_reload_rejects_new_executions_and_concurrent_lifecycle_ops(self):
        """Test reload transitional state rejects new executions and concurrent lifecycle operations."""
        sync_event = threading.Event()
        engine = MockInferenceEngine(sync_event=sync_event)
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://model_base")

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        # Start in-flight execution
        t_exec = threading.Thread(target=lambda: manager.execute("data"))
        t_exec.start()

        for _ in range(50):
            if manager.active_executions > 0:
                break
            time.sleep(0.01)

        reload_started = threading.Event()
        reload_exception = []

        def do_reload():
            try:
                reload_started.set()
                manager.reload("local://model_target", timeout=2.0)
            except Exception as e:
                reload_exception.append(e)

        t_reload = threading.Thread(target=do_reload)
        t_reload.start()

        # Wait until reload enters RELOADING state
        for _ in range(50):
            if manager.state == ExecutionState.RELOADING:
                break
            time.sleep(0.01)
        self.assertEqual(manager.state, ExecutionState.RELOADING)

        # 1. New execute() must be rejected with StateError
        with self.assertRaises(StateError) as exec_err:
            manager.execute("payload while reloading")
        self.assertIn("RELOADING", str(exec_err.exception))

        # 2. Concurrent reload() must be rejected with StateError
        with self.assertRaises(StateError) as reload_err:
            manager.reload("local://another_model")
        self.assertIn("RELOADING", str(reload_err.exception))

        # 3. Concurrent unload() must be rejected with StateError
        with self.assertRaises(StateError) as unload_err:
            manager.unload()
        self.assertIn("RELOADING", str(unload_err.exception))

        # 4. Concurrent shutdown() must be rejected with StateError
        with self.assertRaises(StateError) as shut_err:
            manager.shutdown()
        self.assertIn("RELOADING", str(shut_err.exception))

        # Release in-flight execution to let reload finish
        sync_event.set()
        t_exec.join()
        t_reload.join()

        self.assertEqual(len(reload_exception), 0)
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(manager.model_id, "local://model_target")
        manager.shutdown()

    def test_s2_execution_uses_bound_model_instance_across_pipeline(self):
        """Test execution binds the model instance at execute() call time and doesn't swap mid-flight."""
        model_seen_in_inference = []

        class InspectingInferenceEngine(BaseInferenceEngine):
            def infer(self, model_instance: Any, preprocessed_input: Any, context: ExecutionContext) -> Any:
                model_seen_in_inference.append(model_instance)
                return "output"

        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://model_alpha")
        engine = InspectingInferenceEngine()

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        model_alpha = manager._active_model_instance
        self.assertEqual(model_alpha["loaded_model_source"], "local://model_alpha")

        res = manager.execute("test input")
        self.assertTrue(res.success)
        self.assertEqual(len(model_seen_in_inference), 1)
        self.assertIs(model_seen_in_inference[0], model_alpha)
        manager.shutdown()

    def test_s3_inference_serialized_under_inference_lock(self):
        """Test concurrent executions serialize inference forward passes without model race conditions."""
        class ConcurrencyTrackingEngine(BaseInferenceEngine):
            def __init__(self):
                self.concurrent_counter = 0
                self.max_concurrency = 0
                self._counter_lock = threading.Lock()

            def infer(self, model_instance: Any, preprocessed_input: Any, context: ExecutionContext) -> Any:
                with self._counter_lock:
                    self.concurrent_counter += 1
                    if self.concurrent_counter > self.max_concurrency:
                        self.max_concurrency = self.concurrent_counter

                # Simulate non-instantaneous inference forward pass
                time.sleep(0.02)

                with self._counter_lock:
                    self.concurrent_counter -= 1
                return f"PROCESSED_{preprocessed_input}"

        engine = ConcurrencyTrackingEngine()
        cfg = FrameworkConfig(model_source="local://shared_model")
        manager = ExecutionManager(config=cfg, inference_engine=engine)
        manager.initialize()

        threads = []
        results = [None] * 6

        def worker(idx):
            results[idx] = manager.execute(f"item_{idx}")

        for i in range(6):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All executions must succeed
        for res in results:
            self.assertIsNotNone(res)
            self.assertTrue(res.success)

        # Deterministic verification: inference concurrency must never exceed 1
        self.assertEqual(engine.max_concurrency, 1)
        manager.shutdown()

    def test_s4_unload_drain_timeout_preserves_model_and_state(self):
        """Test unload() drain timeout raises ResourceCleanupError and preserves model and state."""
        sync_event = threading.Event()
        engine = MockInferenceEngine(sync_event=sync_event)
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://critical_model", timeout_seconds=0.05)

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        # Start in-flight execution
        t_exec = threading.Thread(target=lambda: manager.execute("payload"))
        t_exec.start()

        for _ in range(50):
            if manager.active_executions > 0:
                break
            time.sleep(0.01)

        # Unload should fail because execution is still in-flight
        with self.assertRaises(ResourceCleanupError) as cleanup_err:
            manager.unload(timeout=0.05)

        self.assertIn("failed to drain", str(cleanup_err.exception))

        # Verify model was NOT unloaded and state remains READY
        self.assertFalse(loader.unloaded)
        self.assertIsNotNone(manager._active_model_instance)
        self.assertEqual(manager.state, ExecutionState.READY)

        # Release in-flight execution
        sync_event.set()
        t_exec.join()

        # Repeated unload now succeeds cleanly
        manager.unload()
        self.assertTrue(loader.unloaded)
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
        self.assertIsNone(manager._active_model_instance)

        # Idempotent repeated unload is safe
        manager.unload()
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)

    def test_s4_shutdown_drain_timeout_preserves_model_and_state(self):
        """Test shutdown() drain timeout raises ResourceCleanupError and preserves model and executor."""
        sync_event = threading.Event()
        engine = MockInferenceEngine(sync_event=sync_event)
        loader = MockModelLoader()
        cfg = FrameworkConfig(model_source="local://critical_model_shutdown", timeout_seconds=0.05)

        manager = ExecutionManager(config=cfg, model_loader=loader, inference_engine=engine)
        manager.initialize()

        # Start in-flight execution
        t_exec = threading.Thread(target=lambda: manager.execute("payload"))
        t_exec.start()

        for _ in range(50):
            if manager.active_executions > 0:
                break
            time.sleep(0.01)

        # Shutdown should fail because execution is still in-flight
        with self.assertRaises(ResourceCleanupError) as cleanup_err:
            manager.shutdown(timeout=0.05)

        self.assertIn("failed to drain", str(cleanup_err.exception))

        # Verify model was NOT unloaded and manager remains READY
        self.assertFalse(loader.unloaded)
        self.assertIsNotNone(manager._active_model_instance)
        self.assertEqual(manager.state, ExecutionState.READY)

        # Release in-flight execution
        sync_event.set()
        t_exec.join()

        # Repeated shutdown now succeeds cleanly
        manager.shutdown()
        self.assertTrue(loader.unloaded)
        self.assertEqual(manager.state, ExecutionState.STOPPED)
        self.assertIsNone(manager._active_model_instance)

        # Idempotent repeated shutdown is safe
        manager.shutdown()
        self.assertEqual(manager.state, ExecutionState.STOPPED)

    def test_shutdown_invokes_model_loader_close_even_if_already_unloaded(self):
        """Test that ExecutionManager.shutdown() calls model_loader.close() even after unload()."""
        loader = MockModelLoader()

        manager = ExecutionManager(
            config=FrameworkConfig(model_source="local://models/v1"),
            model_loader=loader,
        )
        manager.initialize()
        self.assertEqual(manager.state, ExecutionState.READY)

        # Explicitly unload active model
        manager.unload()
        self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
        self.assertTrue(loader.unloaded)
        self.assertFalse(loader.closed)

        # Stop/shutdown should invoke close() on model_loader
        manager.stop()
        self.assertEqual(manager.state, ExecutionState.STOPPED)
        self.assertTrue(loader.closed)


if __name__ == "__main__":
    unittest.main()

