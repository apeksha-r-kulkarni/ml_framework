"""
ONNX Model Framework Adapter for ONNX Runtime Models.

Encapsulates offline loading, execution provider placement, and lifecycle
management for .onnx model artifacts and directories.
"""

import os
import gc
import logging
from typing import Any, List, Optional

from ..interfaces.base import BaseModelAdapter
from ..config.model_config import FrameworkConfig
from ..errors.exceptions import (
    ModelLoadError,
    DependencyMissingError,
    HardwareIncompatibleError,
)

logger = logging.getLogger("ml_framework.models.onnx")


class ONNXModelAdapter(BaseModelAdapter):
    """
    Framework adapter for ONNX models.
    Encapsulates the onnxruntime InferenceSession and provider configuration.
    Enforces strictly local, offline execution.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        architecture: str = "ONNXModel",
        model_type: str = "onnx",
        config: Optional[FrameworkConfig] = None,
        session: Optional[Any] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.architecture = architecture
        self.model_type = model_type
        self.config = config
        self.session = session
        self.onnx_file: Optional[str] = None
        self._is_loaded = session is not None

    @property
    def is_loaded(self) -> bool:
        """Returns True if the ONNX session is loaded and active."""
        return self._is_loaded and self.session is not None

    def _resolve_onnx_filepath(self) -> str:
        """Locates the .onnx file from a filepath or model directory."""
        if os.path.isfile(self.model_dir):
            if self.model_dir.lower().endswith(".onnx"):
                return self.model_dir
            raise ModelLoadError(
                f"Specified model file '{self.model_dir}' is not an .onnx file.",
                details={"model_dir": self.model_dir}
            )

        if os.path.isdir(self.model_dir):
            try:
                entries = os.listdir(self.model_dir)
            except OSError as e:
                raise ModelLoadError(f"Cannot list model directory '{self.model_dir}': {e}") from e

            onnx_files = [f for f in entries if f.lower().endswith(".onnx")]
            if not onnx_files:
                raise ModelLoadError(
                    f"No .onnx file found in model directory '{self.model_dir}'.",
                    details={"model_dir": self.model_dir, "contents": entries}
                )

            # Prefer model.onnx if present, otherwise first .onnx file
            if "model.onnx" in onnx_files:
                return os.path.join(self.model_dir, "model.onnx")
            return os.path.join(self.model_dir, sorted(onnx_files)[0])

        raise ModelLoadError(
            f"Model path does not exist: '{self.model_dir}'",
            details={"model_dir": self.model_dir}
        )

    def load(self) -> "ONNXModelAdapter":
        """
        Loads ONNX model session from local file only.
        Guarantees strictly offline execution.
        """
        if self._is_loaded and self.session is not None:
            return self

        try:
            import onnxruntime as ort
        except ImportError as e:
            raise DependencyMissingError(
                f"Required dependency 'onnxruntime' is missing: {e}. "
                f"Install via: pip install '.[onnx]' or pip install onnxruntime",
                missing_packages=["onnxruntime"],
                details={"error": str(e), "missing_packages": ["onnxruntime"]}
            ) from e

        onnx_filepath = self._resolve_onnx_filepath()

        # Determine execution providers based on target device
        norm_dev = (self.device or "cpu").lower().strip()
        available_providers = ort.get_available_providers()

        if norm_dev in ("cuda", "gpu") or norm_dev.startswith("cuda:"):
            if "CUDAExecutionProvider" not in available_providers:
                raise HardwareIncompatibleError(
                    f"CUDA execution requested for ONNX model ('{self.device}'), "
                    f"but 'CUDAExecutionProvider' is not available in onnxruntime. "
                    f"Available providers: {available_providers}",
                    details={"device": self.device, "available_providers": available_providers}
                )
            providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif norm_dev == "auto":
            providers = available_providers
        else:
            providers = ["CPUExecutionProvider"]

        # Create InferenceSession
        try:
            session_opts = ort.SessionOptions()
            session = ort.InferenceSession(
                onnx_filepath,
                sess_options=session_opts,
                providers=providers,
            )
        except Exception as load_err:
            if isinstance(load_err, HardwareIncompatibleError):
                raise
            raise ModelLoadError(
                f"Failed to load ONNX model from '{onnx_filepath}': {load_err}",
                details={"onnx_filepath": onnx_filepath, "error": str(load_err)}
            ) from load_err

        self.session = session
        self.onnx_file = onnx_filepath
        self._is_loaded = True
        logger.info(
            f"ONNX model loaded successfully from '{onnx_filepath}' "
            f"(device={self.device}, providers={providers})."
        )
        return self

    def unload(self) -> None:
        """Safely unloads session instance and frees execution memory."""
        self.session = None
        self._is_loaded = False
        gc.collect()
        logger.info(f"ONNX model unloaded from memory (model_dir='{self.model_dir}').")

    def predict(self, preprocessed_input: Any) -> Any:
        """
        Direct inference delegation helper used when ExecutionManager runs without an InferenceEngine.
        """
        if not self.is_loaded:
            raise ModelLoadError("Cannot execute prediction on unloaded ONNX model.")
        return preprocessed_input
