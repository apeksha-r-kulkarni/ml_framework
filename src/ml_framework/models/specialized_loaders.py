"""
Specialized Model Loaders for ML Framework.

Provides dedicated, framework-specific loaders for:
- Speech models (Whisper, Wav2Vec2, HuBERT)
- spaCy pipelines
- Hugging Face / Transformers models
- ONNX Runtime models
"""

import os
import shutil
import tarfile
import zipfile
import tempfile
import logging
from typing import Any, List, Optional
from urllib.parse import urlparse

from ..interfaces.base import BaseModelLoader, BaseEnvironmentManager, BaseModelAdapter
from ..config.model_config import FrameworkConfig, StorageConfig
from ..environment.environment_manager import EnvironmentManager
from ..errors.exceptions import (
    ModelLoadError,
    ModelFormatError,
    StorageError,
)
from .format_detector import FormatDetector
from .speech_adapter import SpeechModelAdapter
from .spacy_adapter import SpacyModelAdapter
from .transformers_adapter import TransformersModelAdapter
from .onnx_adapter import ONNXModelAdapter

logger = logging.getLogger("ml_framework.models.loaders")


class BaseSpecializedModelLoader(BaseModelLoader):
    """
    Base class for specialized model loaders.
    Handles artifact path normalization, remote artifact downloading,
    environment validation, and resource cleanup.
    """

    SUPPORTED_STORAGE_SCHEMES = {"s3", "minio", "seaweedfs"}
    FRAMEWORK_NAME = "generic"

    def __init__(
        self,
        env_manager: Optional[BaseEnvironmentManager] = None,
        storage_config: Optional[StorageConfig] = None,
    ):
        self.env_manager = env_manager if env_manager is not None else EnvironmentManager()
        self.storage_config = storage_config if storage_config is not None else StorageConfig()
        self.format_detector = FormatDetector()
        self._temp_dirs: List[str] = []

    def _clean_path(self, source: str) -> str:
        """Strips local URI schemes and normalizes filesystem paths."""
        path = str(source).strip()
        if path.startswith("file://"):
            return path[7:]
        if path.startswith("local://"):
            return path[8:]
        return path

    def _download_remote_artifact(self, storage_uri: str) -> str:
        """Downloads model artifact archive from remote object storage."""
        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise StorageError(
                f"boto3 is required to download model artifacts from '{storage_uri}'. "
                f"Install via: pip install '.[storage]' or pip install boto3"
            ) from e

        parsed = urlparse(storage_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            raise StorageError(f"Invalid remote storage URI for model artifact: '{storage_uri}'")

        temp_dir = tempfile.mkdtemp(prefix="ml_framework_model_")
        self._temp_dirs.append(temp_dir)

        try:
            boto_cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
            s3 = boto3.client(
                "s3",
                endpoint_url=self.storage_config.endpoint_url,
                aws_access_key_id=self.storage_config.access_key,
                aws_secret_access_key=self.storage_config.secret_key,
                region_name=self.storage_config.region,
                use_ssl=self.storage_config.secure,
                verify=self.storage_config.verify_ssl,
                config=boto_cfg,
            )

            artifact_filename = os.path.basename(key)
            dest_file = os.path.join(temp_dir, artifact_filename)
            logger.info(f"Downloading remote model artifact from '{storage_uri}' to '{dest_file}'...")
            s3.download_file(Bucket=bucket, Key=key, Filename=dest_file)

            if artifact_filename.endswith((".tar.gz", ".tgz")):
                with tarfile.open(dest_file, "r:gz") as tar:
                    tar.extractall(path=temp_dir)
                os.remove(dest_file)
                return temp_dir
            elif artifact_filename.endswith(".tar"):
                with tarfile.open(dest_file, "r:") as tar:
                    tar.extractall(path=temp_dir)
                os.remove(dest_file)
                return temp_dir
            elif artifact_filename.endswith(".zip"):
                with zipfile.ZipFile(dest_file, "r") as z:
                    z.extractall(path=temp_dir)
                os.remove(dest_file)
                return temp_dir
            else:
                return temp_dir

        except Exception as err:
            if temp_dir in self._temp_dirs:
                self._temp_dirs.remove(temp_dir)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(err, StorageError):
                raise
            raise StorageError(
                f"Failed to download remote model artifact from '{storage_uri}': {err}",
                details={"uri": storage_uri, "error": str(err)}
            ) from err

    def load_resolved(
        self,
        local_model_dir: str,
        framework: str,
        architecture: str,
        model_type: str,
        target_device: str,
        config: Optional[FrameworkConfig] = None,
    ) -> BaseModelAdapter:
        """Instantiates and loads the specialized adapter given pre-resolved metadata."""
        raise NotImplementedError("Subclasses must implement load_resolved")

    def load(self, model_source: str, config: Optional[FrameworkConfig] = None) -> BaseModelAdapter:
        """Standalone loading entrypoint for this specialized loader."""
        if not model_source or not str(model_source).strip():
            raise ModelLoadError("Model source cannot be empty or None")

        cfg = config or FrameworkConfig(model_source=model_source)
        source_str = str(model_source).strip()

        parsed = urlparse(source_str)
        is_remote = parsed.scheme.lower() in self.SUPPORTED_STORAGE_SCHEMES

        if is_remote:
            local_model_dir = self._download_remote_artifact(source_str)
        else:
            local_model_dir = self._clean_path(source_str)

        try:
            if not os.path.exists(local_model_dir):
                raise ModelLoadError(
                    f"Local model path does not exist: '{local_model_dir}'",
                    details={"model_source": model_source, "resolved_path": local_model_dir}
                )

            # Detect format and verify compatibility with this specialized loader
            framework, architecture, model_type = self.format_detector.detect(
                local_model_dir, configured_framework=cfg.framework or self.FRAMEWORK_NAME
            )

            # Pre-validate environment & hardware
            if self.env_manager:
                required_deps = self.env_manager.inspect_dependencies(local_model_dir, framework=framework)
                self.env_manager.validate_dependencies(
                    required_deps, context_label=f"model '{source_str}' (framework='{framework}')"
                )
                target_device = self.env_manager.resolve_device(cfg.device)
                self.env_manager.validate_hardware(target_device)
            else:
                target_device = cfg.device

            return self.load_resolved(
                local_model_dir=local_model_dir,
                framework=framework,
                architecture=architecture,
                model_type=model_type,
                target_device=target_device,
                config=cfg,
            )
        except Exception:
            if is_remote and local_model_dir in self._temp_dirs:
                self._temp_dirs.remove(local_model_dir)
                if os.path.exists(local_model_dir):
                    shutil.rmtree(local_model_dir, ignore_errors=True)
            raise

    def unload(self, model_instance: Any) -> None:
        """Safely unloads model instances and frees execution memory."""
        if model_instance is not None:
            if hasattr(model_instance, "unload") and callable(model_instance.unload):
                try:
                    model_instance.unload()
                except Exception as e:
                    logger.warning(f"Error during model instance unload: {e}")
                    raise
        self._cleanup_temp_dirs()

    def _cleanup_temp_dirs(self) -> None:
        """Removes temporary download directories."""
        for d in self._temp_dirs:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary directory '{d}': {e}")
        self._temp_dirs.clear()

    def close(self) -> None:
        """Releases underlying resources and cleans temporary files."""
        self._cleanup_temp_dirs()


class SpeechModelLoader(BaseSpecializedModelLoader):
    """Specialized loader for speech models (Whisper, Wav2Vec2, HuBERT, etc.)."""

    FRAMEWORK_NAME = "speech"

    def load_resolved(
        self,
        local_model_dir: str,
        framework: str,
        architecture: str,
        model_type: str,
        target_device: str,
        config: Optional[FrameworkConfig] = None,
    ) -> SpeechModelAdapter:
        adapter = SpeechModelAdapter(
            model_dir=local_model_dir,
            device=target_device,
            architecture=architecture,
            model_type=model_type,
            config=config,
        )
        adapter.load()
        return adapter


class SpacyModelLoader(BaseSpecializedModelLoader):
    """Specialized loader for spaCy NLP pipelines."""

    FRAMEWORK_NAME = "spacy"

    def load_resolved(
        self,
        local_model_dir: str,
        framework: str,
        architecture: str,
        model_type: str,
        target_device: str,
        config: Optional[FrameworkConfig] = None,
    ) -> SpacyModelAdapter:
        adapter = SpacyModelAdapter(
            model_dir=local_model_dir,
            device=target_device,
            architecture=architecture,
            model_type=model_type,
            config=config,
        )
        adapter.load()
        return adapter


class TransformersModelLoader(BaseSpecializedModelLoader):
    """Specialized loader for Hugging Face / Transformers text models."""

    FRAMEWORK_NAME = "transformers"

    def load_resolved(
        self,
        local_model_dir: str,
        framework: str,
        architecture: str,
        model_type: str,
        target_device: str,
        config: Optional[FrameworkConfig] = None,
    ) -> TransformersModelAdapter:
        adapter = TransformersModelAdapter(
            model_dir=local_model_dir,
            device=target_device,
            architecture=architecture,
            model_type=model_type,
            config=config,
        )
        adapter.load()
        return adapter


class ONNXModelLoader(BaseSpecializedModelLoader):
    """Specialized loader for ONNX runtime models."""

    FRAMEWORK_NAME = "onnx"

    def load_resolved(
        self,
        local_model_dir: str,
        framework: str,
        architecture: str,
        model_type: str,
        target_device: str,
        config: Optional[FrameworkConfig] = None,
    ) -> ONNXModelAdapter:
        adapter = ONNXModelAdapter(
            model_dir=local_model_dir,
            device=target_device,
            architecture=architecture,
            model_type=model_type,
            config=config,
        )
        adapter.load()
        return adapter
