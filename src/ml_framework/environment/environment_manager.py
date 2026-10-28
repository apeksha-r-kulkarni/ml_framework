"""
Environment Manager implementation for dependency inspection, offline environment validation,
and hardware accelerator placement.

Completely decoupled from model loading and inference. Strictly air-gapped safe (zero network calls).
"""

import os
import json
import logging
import importlib.util
from typing import Any, Dict, List, Optional, Set

from ..interfaces.base import BaseEnvironmentManager
from ..config.model_config import FrameworkConfig
from ..errors.exceptions import (
    EnvironmentError,
    DependencyMissingError,
    HardwareIncompatibleError,
)

logger = logging.getLogger("ml_framework.environment")


class EnvironmentManager(BaseEnvironmentManager):
    """
    Validates runtime library dependencies and compute devices.

    Inspects model metadata and configuration to detect required packages
    and reports clear, actionable offline installation instructions
    without downloading or installing packages dynamically.
    """

    FRAMEWORK_DEPENDENCIES: Dict[str, List[str]] = {
        "spacy": ["spacy"],
        "transformers": ["transformers", "torch"],
        "huggingface": ["transformers", "torch"],
        "onnx": ["onnxruntime"],
        "onnxruntime": ["onnxruntime"],
        "speech": ["torch", "transformers", "soundfile", "numpy"],
        "nemo": ["torch", "nemo_toolkit", "soundfile", "numpy"],
        "gliner": ["gliner", "transformers", "torch"],
        "storage": ["boto3"],
        "kafka": ["confluent_kafka"],
    }

    # Canonical mapping of frameworks to pip install extras
    FRAMEWORK_EXTRAS: Dict[str, str] = {
        "spacy": "spacy",
        "transformers": "transformers",
        "huggingface": "transformers",
        "onnx": "onnx",
        "onnxruntime": "onnx",
        "speech": "speech",
        "nemo": "speech",
        "storage": "storage",
        "kafka": "kafka",
    }

    PACKAGE_TO_MODULE_MAP: Dict[str, str] = {
        "confluent-kafka": "confluent_kafka",
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "nemo_toolkit": "nemo",
        "nemo-toolkit": "nemo",
    }

    SPEECH_MODEL_TYPES: Set[str] = {
        "whisper",
        "wav2vec2",
        "hubert",
        "speech_to_text",
        "speech2text",
        "mms",
        "unispeech",
        "unispeech_sat",
        "sew",
        "sew_d",
        "audio_spectrogram_transformer",
        "encodec",
        "seamless_m4t",
        "bark",
        "conformer_ctc",
        "nemo_asr",
        "nemo",
    }

    def __init__(
        self,
        override_packages: Optional[Dict[str, bool]] = None,
        override_device: Optional[str] = None,
        format_detector: Optional[Any] = None,
    ):
        # Allows unit tests to simulate installed or missing packages without altering sys.modules
        self._override_packages = override_packages or {}
        self._override_device = override_device
        self._resolved_device: Optional[str] = None
        self._format_detector = format_detector

    def _get_format_detector(self) -> Any:
        """Returns or lazily creates a FormatDetector instance without circular import issues."""
        if self._format_detector is not None:
            return self._format_detector
        from ..models.format_detector import FormatDetector
        self._format_detector = FormatDetector()
        return self._format_detector

    def is_package_installed(self, package_name: str) -> bool:
        """
        Checks whether a Python package is importable in the local offline environment.
        Strictly local check using importlib.util.find_spec.
        """
        if package_name in self._override_packages:
            return self._override_packages[package_name]

        try:
            module_name = self.PACKAGE_TO_MODULE_MAP.get(
                package_name, package_name.replace("-", "_")
            )
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except Exception:
            return False

    def inspect_dependencies(
        self,
        model_source_or_metadata: Any,
        framework: str = "auto"
    ) -> List[str]:
        """
        Inspects model source directory, path, or metadata dictionary to determine required dependencies.
        Supports:
        - Framework identifier strings ('transformers', 'spacy', 'onnx', 'speech')
        - Model directories containing config.json, meta.json, config.cfg, or .onnx files
        - Standalone .onnx files
        - Speech models (whisper, wav2vec2, preprocessor_config.json) requiring soundfile and numpy
        - Remote storage schemes (s3://, minio://, seaweedfs://) requiring boto3
        - Kafka topic streams (kafka://) requiring confluent_kafka
        """
        required: Set[str] = set()
        norm_framework = framework.lower().strip() if framework else "auto"

        # 1. Framework explicitly declared
        if norm_framework != "auto" and norm_framework in self.FRAMEWORK_DEPENDENCIES:
            required.update(self.FRAMEWORK_DEPENDENCIES[norm_framework])

        # 2. Inspect filesystem path or URI if model_source is a path
        if isinstance(model_source_or_metadata, (str, os.PathLike)):
            path = str(model_source_or_metadata).strip()
            if path.startswith("file://"):
                path = path[7:]

            lower_path = path.lower()

            # Direct .onnx file
            if lower_path.endswith(".onnx"):
                required.update(self.FRAMEWORK_DEPENDENCIES["onnx"])

            # Direct .nemo file
            elif lower_path.endswith(".nemo"):
                required.update(self.FRAMEWORK_DEPENDENCIES["nemo"])

            # Directory inspection
            elif os.path.isdir(path):
                try:
                    detector = self._get_format_detector()
                    detected_fw, _, _ = detector.detect(path, configured_framework=norm_framework)
                    if detected_fw in self.FRAMEWORK_DEPENDENCIES:
                        required.update(self.FRAMEWORK_DEPENDENCIES[detected_fw])
                except Exception as e:
                    logger.debug(f"Format detection during dependency inspection for '{path}': {e}")

            # Remote storage source
            if lower_path.startswith(("s3://", "minio://", "seaweedfs://")):
                required.update(self.FRAMEWORK_DEPENDENCIES["storage"])

            # Kafka source
            if lower_path.startswith("kafka://"):
                required.update(self.FRAMEWORK_DEPENDENCIES["kafka"])

        # 3. Inspect metadata dictionary
        elif isinstance(model_source_or_metadata, dict):
            fw = str(model_source_or_metadata.get("framework", "")).lower()
            if fw in self.FRAMEWORK_DEPENDENCIES:
                required.update(self.FRAMEWORK_DEPENDENCIES[fw])

            m_type = str(model_source_or_metadata.get("model_type", "")).lower()
            if m_type in ("conformer_ctc", "nemo_asr", "nemo") or fw == "nemo":
                required.update(self.FRAMEWORK_DEPENDENCIES["nemo"])
            elif m_type in self.SPEECH_MODEL_TYPES or fw == "speech":
                required.update(self.FRAMEWORK_DEPENDENCIES["speech"])

            format_type = str(model_source_or_metadata.get("format", "") or model_source_or_metadata.get("model_format", "")).lower()
            if format_type == "onnx" or model_source_or_metadata.get("has_onnx"):
                required.update(self.FRAMEWORK_DEPENDENCIES["onnx"])

            if model_source_or_metadata.get("requires_storage"):
                required.update(self.FRAMEWORK_DEPENDENCIES["storage"])

            if model_source_or_metadata.get("requires_kafka"):
                required.update(self.FRAMEWORK_DEPENDENCIES["kafka"])

            extra_deps = model_source_or_metadata.get("dependencies", [])
            if isinstance(extra_deps, (list, tuple, set)):
                for dep in extra_deps:
                    if dep and isinstance(dep, str):
                        required.add(dep.strip())

        return sorted(list(required))

    def validate_dependencies(self, required_packages: List[str], context_label: str = "") -> None:
        """
        Verifies that all required packages are present in the offline environment.
        Raises DependencyMissingError with actionable offline install commands if missing.
        """
        missing = [pkg for pkg in required_packages if not self.is_package_installed(pkg)]

        if missing:
            rec_offline = (
                f"pip install --no-index --find-links=/path/to/offline/wheels {' '.join(missing)}"
            )
            msg = (
                f"Environment validation failed: Missing required dependencies{f' for {context_label}' if context_label else ''}: {missing}. "
                f"Offline installation recommendation: {rec_offline}"
            )
            logger.error(msg)
            raise DependencyMissingError(
                message=msg,
                missing_packages=missing,
                details={
                    "missing_packages": missing,
                    "offline_install_command": rec_offline,
                    "context": context_label,
                }
            )

    def validate_hardware(self, device: str) -> None:
        """
        Verifies that the requested compute hardware accelerator is present and usable.
        Supports: 'cpu', 'auto', 'cuda', 'cuda:<index>', and 'mps'.
        Raises HardwareIncompatibleError if device is unavailable or unsupported.
        """
        if not device or not isinstance(device, str):
            raise HardwareIncompatibleError(f"Device must be a non-empty string, got: {device!r}")

        norm_device = device.lower().strip()

        if norm_device in {"cpu", "auto"}:
            return

        # Check custom override for testing if set
        if self._override_device is not None:
            if norm_device == self._override_device or (norm_device == "cuda" and self._override_device.startswith("cuda")):
                return
            raise HardwareIncompatibleError(
                f"Simulated hardware environment does not support device '{device}'. "
                f"Configured override device: '{self._override_device}'."
            )

        if norm_device == "cuda" or norm_device.startswith("cuda:"):
            if not self.is_package_installed("torch"):
                raise HardwareIncompatibleError(
                    "CUDA device requested, but PyTorch ('torch') is not installed to verify or use CUDA. "
                    "Install PyTorch with CUDA support or set device='cpu'."
                )
            try:
                import torch
                if not torch.cuda.is_available():
                    raise HardwareIncompatibleError(
                        "CUDA acceleration was requested (device='cuda'), but no CUDA-compatible GPU is available. "
                        "Fallback to device='cpu' or verify GPU drivers."
                    )

                # If specific device index requested (e.g. cuda:0, cuda:1)
                if ":" in norm_device:
                    idx_str = norm_device.split(":", 1)[1]
                    try:
                        device_idx = int(idx_str)
                    except ValueError:
                        raise HardwareIncompatibleError(
                            f"Invalid CUDA device index '{idx_str}' in device string '{device}'. "
                            "Expected integer index, e.g. 'cuda:0'."
                        )
                    device_count = torch.cuda.device_count()
                    if device_idx < 0 or device_idx >= device_count:
                        raise HardwareIncompatibleError(
                            f"CUDA device index {device_idx} requested, but only {device_count} "
                            f"CUDA device(s) are available (indices 0 to {max(0, device_count - 1)})."
                        )
            except Exception as e:
                if isinstance(e, HardwareIncompatibleError):
                    raise
                raise HardwareIncompatibleError(f"Failed to query CUDA availability: {e}") from e

        elif norm_device == "mps":
            if not self.is_package_installed("torch"):
                raise HardwareIncompatibleError("MPS device requested, but PyTorch is not installed.")
            try:
                import torch
                if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                    raise HardwareIncompatibleError(
                        "Apple Silicon MPS acceleration was requested (device='mps'), but MPS is not available."
                    )
            except Exception as e:
                if isinstance(e, HardwareIncompatibleError):
                    raise
                raise HardwareIncompatibleError(f"Failed to query MPS availability: {e}") from e

        else:
            raise HardwareIncompatibleError(
                f"Unsupported device type: '{device}'. Supported devices: 'auto', 'cpu', 'cuda', 'cuda:<index>', 'mps'."
            )

    def resolve_device(self, requested_device: Optional[str] = "auto") -> str:
        """
        Resolves the actual hardware device to be used based on preference and system availability.
        """
        if self._override_device is not None:
            return self._override_device

        device_str = (requested_device or "auto").lower().strip()

        if device_str == "cpu":
            return "cpu"

        if device_str in ("cuda", "mps") or device_str.startswith("cuda:"):
            self.validate_hardware(device_str)
            return device_str

        if device_str == "auto":
            if self.is_package_installed("torch"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        return "cuda"
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        return "mps"
                except Exception:
                    pass
            return "cpu"

        self.validate_hardware(device_str)
        return device_str

    def get_device(self, requested_device: Optional[str] = None) -> str:
        """
        Returns the resolved compute device ('cuda', 'mps', or 'cpu').
        If requested_device is provided, resolves it.
        Otherwise returns the device resolved during validate_environment(),
        or falls back to auto-detecting the best available hardware accelerator.
        """
        if requested_device is not None:
            return self.resolve_device(requested_device)
        if self._resolved_device is not None:
            return self._resolved_device
        return self.resolve_device("auto")

    def validate_environment(
        self,
        config: FrameworkConfig,
        model_metadata: Optional[Any] = None
    ) -> bool:
        """
        Full environment validation routine:
        1. Determines and validates required library dependencies.
        2. Validates hardware device compatibility and resolves execution device.
        """
        source = config.model_source or model_metadata
        required_deps = self.inspect_dependencies(source, framework=config.framework)

        # Check storage dependencies if storage endpoint is configured or source uses storage
        if (
            (config.storage_config and config.storage_config.endpoint_url)
            or (config.model_source and str(config.model_source).lower().startswith(("s3://", "minio://", "seaweedfs://")))
            or (config.input_source and str(config.input_source).lower().startswith(("s3://", "minio://", "seaweedfs://")))
            or (config.input_config and config.input_config.source and str(config.input_config.source).lower().startswith(("s3://", "minio://", "seaweedfs://")))
            or (config.input_config and config.input_config.source_type == "object_storage")
        ):
            if "boto3" not in required_deps:
                required_deps.append("boto3")

        # Check Kafka dependency if input_config uses Kafka
        if (
            (config.input_config and config.input_config.kafka_topic)
            or (config.input_config and config.input_config.source and str(config.input_config.source).lower().startswith("kafka://"))
            or (config.input_source and str(config.input_source).lower().startswith("kafka://"))
            or (config.input_config and config.input_config.source_type == "kafka")
        ):
            if "confluent_kafka" not in required_deps:
                required_deps.append("confluent_kafka")

        # Validate library packages
        is_triton_served = bool(
            config.triton_config and config.triton_config.model_name
        ) or config.framework == "triton"
        if is_triton_served:
            backend_only_deps = {"nemo_toolkit", "ctranslate2", "speechbrain"}
            required_deps = [d for d in required_deps if d not in backend_only_deps]

        self.validate_dependencies(
            required_deps,
            context_label=f"framework='{config.framework}'"
        )

        # Validate hardware device and record resolved device
        self.validate_hardware(config.device)
        self._resolved_device = self.resolve_device(config.device)

        logger.info(
            f"Environment validation passed (resolved_device='{self._resolved_device}', deps={required_deps})."
        )
        return True

    def close(self) -> None:
        """Releases any cached environment resources (no-op for stateless manager)."""
        pass
