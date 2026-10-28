"""
NVIDIA Triton Inference Engine for the ML Framework.

Provides the exclusive production inference execution engine, delegating model forward pass
execution to NVIDIA Triton Inference Server (Option B: direct NeMo Conformer CTC serving via Triton Python backend).
Implements BaseInferenceEngine.
"""

import time
import logging
from typing import Any, Dict, Optional, Union

from ..interfaces.base import BaseInferenceEngine
from ..config.model_config import TritonConfig, FrameworkConfig
from ..results.execution_result import ExecutionContext
from ..errors.exceptions import InferenceError, ConfigurationError
from .triton_client import TritonClient
from .tensor_mapper import TensorMapper

logger = logging.getLogger("ml_framework.inference.triton_engine")


class TritonInferenceEngine(BaseInferenceEngine):
    """
    Production inference engine that delegates model execution exclusively to NVIDIA Triton Inference Server.
    Eliminates local model adapter predictions in production pipelines.
    """

    def __init__(
        self,
        config: Optional[TritonConfig] = None,
        client: Optional[TritonClient] = None,
        tensor_mapper: Optional[TensorMapper] = None,
    ):
        self.config = config if config is not None else TritonConfig()
        self.client = client if client is not None else TritonClient(
            endpoint_url=self.config.endpoint_url,
            timeout_seconds=self.config.timeout_seconds,
            max_retries=self.config.max_retries,
            retry_interval_seconds=self.config.retry_interval_seconds,
        )
        self.tensor_mapper = tensor_mapper if tensor_mapper is not None else TensorMapper(
            input_tensor_map=self.config.input_tensor_map,
            output_tensor_map=self.config.output_tensor_map,
        )

    def verify_readiness(
        self,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> bool:
        """
        Validates that NVIDIA Triton Inference Server is running and the specified model is READY.

        Raises:
            InferenceError: If Triton server is not reachable or the model is not ready.
        """
        target_model = model_name or self.config.model_name
        target_version = model_version if model_version is not None else self.config.model_version

        # 1. Check server health
        if not self.client.is_server_ready():
            raise InferenceError(
                f"NVIDIA Triton Inference Server at '{self.config.endpoint_url}' is not ready or unreachable. "
                f"Please verify that the Triton container is running and healthy.",
                details={"endpoint_url": self.config.endpoint_url}
            )

        # 2. Check model readiness if target model is specified
        if target_model:
            if not self.client.is_model_ready(target_model, target_version):
                raise InferenceError(
                    f"Model '{target_model}' (version: '{target_version or 'latest'}') is not READY on Triton at "
                    f"'{self.config.endpoint_url}'. Verify that the model is loaded in the Triton model repository.",
                    details={
                        "model_name": target_model,
                        "model_version": target_version,
                        "endpoint_url": self.config.endpoint_url,
                    }
                )

        logger.info(f"Triton server and model '{target_model or 'unspecified'}' readiness successfully verified.")
        return True

    def resolve_model_name(
        self,
        model: Any,
        context: Optional[ExecutionContext] = None
    ) -> str:
        """
        Resolves the Triton model identifier from execution context, model instance, or configuration.
        """
        # 1. Check ExecutionContext metadata
        if context and context.metadata:
            if context.metadata.get("triton_model_name"):
                return str(context.metadata["triton_model_name"])
            if context.metadata.get("model_name"):
                return str(context.metadata["model_name"])

        # 2. Check engine config
        if self.config.model_name:
            return self.config.model_name

        # 3. Check model instance attributes or dictionary keys
        if model is not None:
            if isinstance(model, dict):
                for k in ("model_name", "name", "architecture", "model_type"):
                    v = model.get(k)
                    if v and isinstance(v, str) and v.strip():
                        return v.strip()
            for attr in ("model_name", "name", "architecture", "model_type"):
                val = getattr(model, attr, None)
                if val and isinstance(val, str) and val.strip():
                    return val.strip()

        # 4. Check context model_id
        if context and context.model_id and context.model_id != "unspecified":
            # Strip source prefixes if present
            mid = context.model_id.split("://")[-1].split("/")[-1].split(":")[0]
            if mid:
                return mid

        raise InferenceError(
            "Unable to resolve Triton model name for inference. "
            "Please configure triton_config.model_name or provide model_name in FrameworkConfig."
        )

    def infer(
        self,
        model: Any,
        preprocessed_input: Any,
        context: ExecutionContext
    ) -> Any:
        """
        Executes inference exclusively through NVIDIA Triton Inference Server.

        Args:
            model: Bound model metadata or adapter instance (for naming context only).
            preprocessed_input: Cleaned and structured input data from InputProcessor.
            context: Execution context containing correlation ID and telemetry.

        Returns:
            Normalized inference result object or dict adhering to framework contracts.
        """
        model_name = self.resolve_model_name(model, context)
        model_version = self.config.model_version

        # 1. Map input domain object to Triton KServe v2 tensor schemas adhering to task contract
        inputs, outputs = self.tensor_mapper.build_request(
            preprocessed_input=preprocessed_input,
            model_name=model_name,
            model_metadata=model if isinstance(model, dict) else None,
        )

        # 2. Execute inference call over Triton HTTP REST API
        t0 = time.perf_counter()
        raw_response = self.client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            model_version=model_version,
            timeout=self.config.timeout_seconds,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Decode Triton response tensors into standardized framework output
        task = model.get("task", "auto") if isinstance(model, dict) else "auto"
        parsed_output = self.tensor_mapper.parse_response(raw_response, task=task)

        # 4. Record execution metadata in context
        if context and context.metadata is not None:
            context.metadata["triton_endpoint"] = self.config.endpoint_url
            context.metadata["triton_model_name"] = model_name
            context.metadata["triton_model_version"] = model_version or "latest"
            context.metadata["triton_inference_latency_ms"] = round(latency_ms, 2)

        return parsed_output
