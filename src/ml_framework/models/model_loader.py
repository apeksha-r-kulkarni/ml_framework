"""
ModelLoader implementation for the ML Framework.

Provides generic model artifact resolution, format detection, environment validation,
and framework adapter instantiation according to the Model Loading Sequence.
Supports extensible specialized loaders for speech, spaCy, Transformers, and ONNX.
"""

import os
import shutil
import tarfile
import zipfile
import tempfile
import logging
from typing import Any, Dict, List, Optional, Type, Tuple
from urllib.parse import urlparse

from ..interfaces.base import BaseModelLoader, BaseEnvironmentManager, BaseModelAdapter
from ..config.model_config import FrameworkConfig, StorageConfig, TritonConfig
from ..environment.environment_manager import EnvironmentManager
from ..errors.exceptions import (
    ModelLoadError,
    ModelFormatError,
    StorageError,
    DependencyMissingError,
)
from .format_detector import FormatDetector
from .speech_adapter import SpeechModelAdapter
from .spacy_adapter import SpacyModelAdapter
from .transformers_adapter import TransformersModelAdapter
from .onnx_adapter import ONNXModelAdapter
from .specialized_loaders import (
    BaseSpecializedModelLoader,
    SpeechModelLoader,
    SpacyModelLoader,
    TransformersModelLoader,
    ONNXModelLoader,
)

logger = logging.getLogger("ml_framework.models.loader")


def _classify_boto_error(
    err: Exception,
    storage_uri: str,
    bucket: str,
    key: str,
    endpoint: Optional[str] = None
) -> StorageError:
    """
    Classifies low-level boto3 / botocore exceptions into clear, actionable StorageError instances.
    Distinguishes authentication failures, missing objects/buckets, and connectivity failures.
    Ensures credentials are never printed or leaked.
    """
    if isinstance(err, StorageError):
        return err

    err_code = ""
    if hasattr(err, "response") and isinstance(err.response, dict):
        err_code = str(err.response.get("Error", {}).get("Code", ""))
    err_msg = str(err)
    err_type = type(err).__name__

    # 1. Authentication / Permission errors
    auth_codes = {"403", "AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "AuthFailure", "InvalidToken", "ExpiredToken"}
    if (
        err_code in auth_codes
        or any(w in err_msg.lower() for w in ("access denied", "accessdenied", "forbidden", "invalidaccesskeyid", "signaturedoesnotmatch", "credential"))
    ):
        return StorageError(
            f"Authentication failed for MinIO storage at '{endpoint or 'unspecified'}': {err_msg}. "
            f"Please verify MINIO_ACCESS_KEY and MINIO_SECRET_KEY.",
            details={"error_code": err_code, "bucket": bucket, "uri": storage_uri, "endpoint": endpoint}
        )

    # 2. Missing object or bucket errors
    not_found_codes = {"404", "NoSuchKey", "NoSuchBucket", "ResourceNotFoundException", "NotFound"}
    if (
        err_code in not_found_codes
        or any(w in err_msg.lower() for w in ("nosuchkey", "nosuchbucket", "not found in storage", "does not exist"))
    ):
        return StorageError(
            f"Model artifact not found in storage: '{storage_uri}' (bucket='{bucket}', key='{key}')",
            details={"error_code": err_code, "bucket": bucket, "key": key, "uri": storage_uri}
        )

    # 3. Network / Connection errors
    conn_types = {"endpointconnectionerror", "connecttimeouterror", "connectionerror", "readtimeouterror", "maxretryerror", "newconnectionerror"}
    if (
        err_type.lower() in conn_types
        or any(w in err_msg.lower() for w in ("connection refused", "could not connect", "failed to connect", "connection reset", "timed out", "unreachable"))
    ):
        return StorageError(
            f"Failed to connect to MinIO storage endpoint at '{endpoint or 'unspecified'}': {err_msg}. "
            f"Please verify that MinIO is running and network connectivity is available.",
            details={"endpoint": endpoint, "uri": storage_uri, "error": err_msg}
        )

    # 4. General Storage Error
    return StorageError(
        f"Failed to download remote model artifact from '{storage_uri}': {err_msg}",
        details={"uri": storage_uri, "bucket": bucket, "key": key, "error": err_msg}
    )


class ModelLoader(BaseModelLoader):
    """
    Generic model loader and adapter resolver.

    1. Resolves model source (local path or S3/MinIO/SeaweedFS storage URI).
    2. Uses FormatDetector to identify model format and architecture.
    3. Uses EnvironmentManager to pre-validate dependencies and target device.
    4. Selects and dispatches to the specialized loader (or adapter) for the detected framework.
    5. Safely unloads model instances and cleans up temporary resources.
    """

    SUPPORTED_STORAGE_SCHEMES = {"s3", "minio", "seaweedfs"}

    DEFAULT_LOADERS: Dict[str, Type[BaseSpecializedModelLoader]] = {
        "speech": SpeechModelLoader,
        "nemo": SpeechModelLoader,
        "transformers": TransformersModelLoader,
        "huggingface": TransformersModelLoader,
        "spacy": SpacyModelLoader,
        "onnx": ONNXModelLoader,
        "onnxruntime": ONNXModelLoader,
    }

    DEFAULT_ADAPTERS: Dict[str, Type[BaseModelAdapter]] = {
        "speech": SpeechModelAdapter,
        "nemo": SpeechModelAdapter,
        "transformers": TransformersModelAdapter,
        "huggingface": TransformersModelAdapter,
        "spacy": SpacyModelAdapter,
        "onnx": ONNXModelAdapter,
        "onnxruntime": ONNXModelAdapter,
    }

    def __init__(
        self,
        env_manager: Optional[BaseEnvironmentManager] = None,
        storage_config: Optional[StorageConfig] = None,
        loader_registry: Optional[Dict[str, Type[BaseModelLoader]]] = None,
        adapter_registry: Optional[Dict[str, Type[BaseModelAdapter]]] = None,
    ):
        self.env_manager = env_manager if env_manager is not None else EnvironmentManager()
        self.storage_config = storage_config if storage_config is not None else StorageConfig()
        self.format_detector = FormatDetector()
        self._temp_dirs: List[str] = []
        self._loader_registry: Dict[str, Type[BaseModelLoader]] = dict(self.DEFAULT_LOADERS)
        if loader_registry:
            self._loader_registry.update(loader_registry)
        self._adapter_registry: Dict[str, Type[BaseModelAdapter]] = dict(self.DEFAULT_ADAPTERS)
        if adapter_registry:
            self._adapter_registry.update(adapter_registry)

    def register_loader(self, framework: str, loader_cls: Type[BaseModelLoader]) -> None:
        """Registers or overrides a specialized loader class for a given framework."""
        self._loader_registry[framework.lower().strip()] = loader_cls

    def register_adapter(self, framework: str, adapter_cls: Type[BaseModelAdapter]) -> None:
        """Registers or overrides a specialized adapter class for a given framework."""
        self._adapter_registry[framework.lower().strip()] = adapter_cls

    def get_supported_frameworks(self) -> List[str]:
        """Returns the sorted list of supported framework keys."""
        all_keys = set(self._loader_registry.keys()) | set(self._adapter_registry.keys())
        return sorted(list(all_keys))

    def get_loader(self, framework: str) -> Optional[BaseModelLoader]:
        """Instantiates and returns the specialized loader for a framework if registered."""
        norm = framework.lower().strip()
        loader_cls = self._loader_registry.get(norm)
        if loader_cls is not None:
            return loader_cls(env_manager=self.env_manager, storage_config=self.storage_config)
        return None

    def _clean_path(self, source: str) -> str:
        """Strips local URI schemes and normalizes filesystem paths."""
        path = str(source).strip()
        if path.startswith("file://"):
            return path[7:]
        if path.startswith("local://"):
            return path[8:]
        return path

    def _download_remote_artifact(self, storage_uri: str) -> str:
        """
        Downloads a model artifact archive or directory from S3 / MinIO / SeaweedFS
        into a dedicated temporary staging directory.
        """
        parsed = urlparse(storage_uri)
        scheme = parsed.scheme.lower()
        if scheme not in self.SUPPORTED_STORAGE_SCHEMES:
            raise StorageError(
                f"Unsupported remote storage scheme '{scheme}' in URI '{storage_uri}'. "
                f"Supported schemes: {sorted(list(self.SUPPORTED_STORAGE_SCHEMES))}"
            )

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise StorageError(f"Missing bucket name in remote storage URI: '{storage_uri}'")
        if not key:
            raise StorageError(f"Missing object key or prefix in remote storage URI: '{storage_uri}'")

        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise StorageError(
                f"boto3 is required to download model artifacts from '{storage_uri}'. "
                f"Install via: pip install '.[storage]' or pip install boto3"
            ) from e

        storage_cfg = self.storage_config
        if not storage_cfg.endpoint_url:
            storage_cfg = StorageConfig.from_env()

        download_base = storage_cfg.download_dir
        if download_base:
            os.makedirs(download_base, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="ml_framework_model_", dir=download_base)
        else:
            temp_dir = tempfile.mkdtemp(prefix="ml_framework_model_")
        self._temp_dirs.append(temp_dir)

        try:
            from boto3.s3.transfer import TransferConfig
            transfer_cfg = TransferConfig(
                max_concurrency=4,
                multipart_chunksize=16 * 1024 * 1024,
                num_download_attempts=10,
            )
            boto_cfg = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 10, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=120,
            )
            s3 = boto3.client(
                "s3",
                endpoint_url=storage_cfg.endpoint_url,
                aws_access_key_id=storage_cfg.access_key,
                aws_secret_access_key=storage_cfg.secret_key,
                region_name=storage_cfg.region,
                use_ssl=storage_cfg.secure,
                verify=storage_cfg.verify_ssl,
                config=boto_cfg,
            )

            norm_key = key.rstrip("/")
            dir_prefix = norm_key + "/"

            paginator = s3.get_paginator("list_objects_v2")
            dir_objects = []
            try:
                for page in paginator.paginate(Bucket=bucket, Prefix=dir_prefix):
                    for obj in page.get("Contents", []):
                        dir_objects.append(obj)
            except Exception as list_err:
                raise _classify_boto_error(list_err, storage_uri, bucket, dir_prefix, storage_cfg.endpoint_url) from list_err

            if dir_objects:
                downloaded_count = 0
                for obj in dir_objects:
                    obj_key = obj["Key"]
                    if obj_key.endswith("/"):
                        continue
                    rel_path = obj_key[len(dir_prefix):]
                    if not rel_path:
                        continue
                    dest_file = os.path.join(temp_dir, rel_path)
                    os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                    logger.info(f"Downloading model artifact file '{obj_key}' to '{dest_file}'...")
                    _s3_dl = lambda b, k, fn: s3.download_file(Bucket=b, Key=k, Filename=fn, Config=transfer_cfg) if transfer_cfg else s3.download_file(Bucket=b, Key=k, Filename=fn)
                    try:
                        _s3_dl(bucket, obj_key, dest_file)
                    except TypeError:
                        s3.download_file(Bucket=bucket, Key=obj_key, Filename=dest_file)
                    downloaded_count += 1

                if downloaded_count == 0:
                    raise StorageError(
                        f"No model artifact files found under prefix '{storage_uri}' (bucket='{bucket}', prefix='{dir_prefix}')",
                        details={"uri": storage_uri, "bucket": bucket, "prefix": dir_prefix}
                    )
            else:
                exact_objects = []
                try:
                    for page in paginator.paginate(Bucket=bucket, Prefix=norm_key):
                        for obj in page.get("Contents", []):
                            exact_objects.append(obj)
                except Exception as list_err:
                    raise _classify_boto_error(list_err, storage_uri, bucket, norm_key, storage_cfg.endpoint_url) from list_err

                if not exact_objects:
                    raise StorageError(
                        f"Model artifact not found in storage: '{storage_uri}' (bucket='{bucket}', key='{key}')",
                        details={"uri": storage_uri, "bucket": bucket, "key": key}
                    )

                is_single_file = len(exact_objects) == 1 and exact_objects[0]["Key"] == norm_key

                if is_single_file:
                    artifact_filename = os.path.basename(norm_key)
                    dest_file = os.path.join(temp_dir, artifact_filename)
                    remote_size = exact_objects[0].get("Size", 0)

                    cached_file = None
                    if download_base and os.path.isdir(download_base):
                        for root, _, files in os.walk(download_base):
                            if artifact_filename in files:
                                candidate = os.path.join(root, artifact_filename)
                                if remote_size > 0 and os.path.getsize(candidate) == remote_size:
                                    cached_file = candidate
                                    break
                    if cached_file is None and os.path.isfile(f"/tmp/poc_nemo_model/{artifact_filename}"):
                        candidate = f"/tmp/poc_nemo_model/{artifact_filename}"
                        if remote_size > 0 and os.path.getsize(candidate) == remote_size:
                            cached_file = candidate

                    if cached_file:
                        logger.info(f"Reusing verified cached model artifact from '{cached_file}' ({remote_size} bytes).")
                        shutil.copy2(cached_file, dest_file)
                    else:
                        logger.info(f"Downloading remote model artifact from '{storage_uri}' to '{dest_file}'...")
                        try:
                            if transfer_cfg is not None:
                                s3.download_file(Bucket=bucket, Key=norm_key, Filename=dest_file, Config=transfer_cfg)
                            else:
                                s3.download_file(Bucket=bucket, Key=norm_key, Filename=dest_file)
                        except TypeError:
                            s3.download_file(Bucket=bucket, Key=norm_key, Filename=dest_file)

                    # Unpack if archive
                    if artifact_filename.endswith((".tar.gz", ".tgz")):
                        with tarfile.open(dest_file, "r:gz") as tar:
                            tar.extractall(path=temp_dir)
                        os.remove(dest_file)
                    elif artifact_filename.endswith(".tar"):
                        with tarfile.open(dest_file, "r:") as tar:
                            tar.extractall(path=temp_dir)
                        os.remove(dest_file)
                    elif artifact_filename.endswith(".zip"):
                        with zipfile.ZipFile(dest_file, "r") as z:
                            z.extractall(path=temp_dir)
                        os.remove(dest_file)
                    elif artifact_filename.endswith(".nemo"):
                        # Preserve native .nemo archive intact for NeMo / Triton Python backend serving
                        pass
                else:
                    prefix_to_strip = dir_prefix if any(o["Key"].startswith(dir_prefix) for o in exact_objects) else (norm_key + "/")
                    downloaded_count = 0
                    for obj in exact_objects:
                        obj_key = obj["Key"]
                        if obj_key.endswith("/"):
                            continue
                        if obj_key.startswith(prefix_to_strip):
                            rel_path = obj_key[len(prefix_to_strip):]
                        else:
                            rel_path = os.path.basename(obj_key)
                        if not rel_path:
                            continue
                        dest_file = os.path.join(temp_dir, rel_path)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        logger.info(f"Downloading model artifact file '{obj_key}' to '{dest_file}'...")
                        try:
                            if transfer_cfg is not None:
                                s3.download_file(Bucket=bucket, Key=obj_key, Filename=dest_file, Config=transfer_cfg)
                            else:
                                s3.download_file(Bucket=bucket, Key=obj_key, Filename=dest_file)
                        except TypeError:
                            s3.download_file(Bucket=bucket, Key=obj_key, Filename=dest_file)
                        downloaded_count += 1

                    if downloaded_count == 0:
                        raise StorageError(
                            f"No model artifact files found under prefix '{storage_uri}' (bucket='{bucket}', prefix='{norm_key}')",
                            details={"uri": storage_uri, "bucket": bucket, "prefix": norm_key}
                        )

            # Check if archive unpacked into a single nested subdirectory, or single file (.nemo, .onnx)
            entries = [e for e in os.listdir(temp_dir) if not e.startswith(".")]
            if len(entries) == 1:
                single_path = os.path.join(temp_dir, entries[0])
                if os.path.isdir(single_path):
                    return single_path
                if entries[0].endswith((".nemo", ".onnx")):
                    return single_path
            return temp_dir

        except Exception as err:
            if temp_dir in self._temp_dirs:
                self._temp_dirs.remove(temp_dir)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
            if isinstance(err, (StorageError, ModelLoadError, ModelFormatError)):
                raise
            raise _classify_boto_error(err, storage_uri, bucket, key, storage_cfg.endpoint_url) from err

    def load(self, model_source: str, config: Optional[FrameworkConfig] = None) -> Any:
        """
        Resolves, downloads (if remote), detects format, validates environment,
        and instantiates the concrete model adapter via specialized loader dispatch.
        """
        if not model_source or not str(model_source).strip():
            raise ModelLoadError("Model source cannot be empty or None")

        cfg = config or FrameworkConfig(model_source=model_source)
        source_str = str(model_source).strip()

        # 1. Resolve path (local or remote)
        parsed = urlparse(source_str)
        scheme = parsed.scheme.lower()
        is_remote = scheme in self.SUPPORTED_STORAGE_SCHEMES

        if is_remote:
            local_model_dir = self._download_remote_artifact(source_str)
        else:
            local_model_dir = self._clean_path(source_str)

        try:
            # 2. Verify local path exists
            if not os.path.exists(local_model_dir):
                raise ModelLoadError(
                    f"Local model path does not exist: '{local_model_dir}'",
                    details={"model_source": model_source, "resolved_path": local_model_dir}
                )

            # 3. Detect model format and architecture
            framework, architecture, model_type = self.format_detector.detect(
                local_model_dir, configured_framework=cfg.framework
            )

            # 4. Pre-validate environment & hardware placement
            if self.env_manager:
                required_deps = self.env_manager.inspect_dependencies(local_model_dir, framework=framework)
                self.env_manager.validate_dependencies(
                    required_deps, context_label=f"model '{source_str}' (framework='{framework}')"
                )
                target_device = self.env_manager.resolve_device(cfg.device)
                self.env_manager.validate_hardware(target_device)
            else:
                target_device = cfg.device

            # 5. Dispatch to specialized loader or adapter
            norm_fw = framework.lower().strip()
            loader_cls = self._loader_registry.get(norm_fw)

            if loader_cls is not None:
                loader = loader_cls(env_manager=self.env_manager, storage_config=self.storage_config)
                if hasattr(loader, "load_resolved") and callable(loader.load_resolved):
                    return loader.load_resolved(
                        local_model_dir=local_model_dir,
                        framework=framework,
                        architecture=architecture,
                        model_type=model_type,
                        target_device=target_device,
                        config=cfg,
                    )
                else:
                    return loader.load(local_model_dir, config=cfg)

            # Fallback to direct adapter instantiation from adapter registry
            adapter_cls = self._adapter_registry.get(norm_fw)
            if adapter_cls is not None:
                adapter = adapter_cls(
                    model_dir=local_model_dir,
                    device=target_device,
                    architecture=architecture,
                    model_type=model_type,
                    config=cfg,
                )
                adapter.load()
                return adapter

            raise ModelFormatError(
                f"Model framework '{framework}' is not supported. "
                f"Currently supported frameworks: {self.get_supported_frameworks()}. "
                f"Detected: architecture='{architecture}', model_type='{model_type}'.",
                details={
                    "framework": framework,
                    "architecture": architecture,
                    "model_type": model_type,
                    "supported_frameworks": self.get_supported_frameworks(),
                }
            )
        except Exception:
            if is_remote:
                self._cleanup_temp_dirs()
            raise

    def validate_backend_compatibility(
        self,
        local_path: str,
        backend: str = "python",
        model_name: Optional[str] = None,
        configured_framework: str = "auto",
    ) -> Tuple[str, str, str]:
        """
        Validates that the downloaded or local model artifact format is compatible with
        the Triton server's configured backend (e.g. NeMo Python backend vs ONNX runtime).

        Args:
            local_path: Path to the local model directory or artifact file.
            backend: Configured Triton backend ("python", "onnx", "onnxruntime", etc.).
            model_name: Optional model identifier for error details.
            configured_framework: Optional framework override.

        Returns:
            Tuple of (detected_framework, architecture, model_type).

        Raises:
            ModelFormatError: If the model format is incompatible with the Triton backend.
        """
        fw, arch, mtype = self.format_detector.detect(
            local_path, configured_framework=configured_framework
        )
        norm_backend = (backend or "python").lower().strip()

        # Check for formats requiring container dependencies absent in active Triton container
        if fw == "ctranslate2":
            raise DependencyMissingError(
                f"Model format 'ctranslate2' ({mtype}) at '{local_path}' requires the 'ctranslate2' engine and Triton CTranslate2 backend, "
                f"which are not installed in the active Triton container.",
                details={"framework": fw, "architecture": arch, "model_type": mtype, "required_package": "ctranslate2"}
            )

        if fw == "deepfilternet":
            raise DependencyMissingError(
                f"Model format 'deepfilternet' ({mtype}) at '{local_path}' requires the 'deepfilternet' package and Triton audio processing backend, "
                f"which are not installed in the active Triton container.",
                details={"framework": fw, "architecture": arch, "model_type": mtype, "required_package": "deepfilternet"}
            )

        if fw == "speechbrain":
            raise DependencyMissingError(
                f"Model format 'speechbrain' ({mtype}) at '{local_path}' requires the 'speechbrain' package and Triton backend, "
                f"which are not installed in the active Triton container.",
                details={"framework": fw, "architecture": arch, "model_type": mtype, "required_package": "speechbrain"}
            )

        # 1. NeMo Python backend validation
        if norm_backend in ("python", "nemo", "nemo_conformer_python", "conformer_python"):
            is_nemo_file = os.path.isfile(local_path) and local_path.lower().endswith(".nemo")
            is_nemo_dir = os.path.isdir(local_path) and any(
                f.lower().endswith(".nemo") or f == "model_config.yaml"
                for f in os.listdir(local_path)
            )

            if fw not in ("nemo", "speech") or not (is_nemo_file or is_nemo_dir):
                raise ModelFormatError(
                    f"Model format '{fw}' (architecture: '{arch}', type: '{mtype}') at '{local_path}' "
                    f"is incompatible with Triton backend '{backend}'. The Triton NeMo Python backend "
                    f"requires a native NeMo model checkpoint (.nemo), but received a '{fw}' artifact.",
                    details={
                        "detected_framework": fw,
                        "architecture": arch,
                        "model_type": mtype,
                        "triton_backend": backend,
                        "model_name": model_name,
                        "resolved_path": local_path,
                    }
                )

            # NeMo LID models (EncDecSpeakerLabelModel) cannot run through ASR Conformer CTC backend
            if arch == "EncDecSpeakerLabelModel" and (
                norm_backend in ("nemo_conformer_python", "conformer_python")
                or (norm_backend == "python" and (not model_name or "conformer" in str(model_name).lower()))
            ):
                raise ModelFormatError(
                    f"NeMo model architecture '{arch}' (Language Identification) at '{local_path}' "
                    f"is incompatible with Triton backend '{backend}' (configured for ASR EncDecCTCModelBPE). "
                    f"Serving this model requires a dedicated Triton LID backend.",
                    details={
                        "architecture": arch,
                        "model_type": mtype,
                        "triton_backend": backend,
                        "model_name": model_name,
                    }
                )

        # 2. ONNX backend validation
        elif norm_backend in ("onnx", "onnxruntime"):
            is_onnx_file = os.path.isfile(local_path) and local_path.lower().endswith(".onnx")
            is_onnx_dir = os.path.isdir(local_path) and any(
                f.lower().endswith(".onnx") for f in os.listdir(local_path)
            )
            if fw != "onnx" and not (is_onnx_file or is_onnx_dir):
                raise ModelFormatError(
                    f"Model format '{fw}' (architecture: '{arch}', type: '{mtype}') at '{local_path}' "
                    f"is incompatible with Triton backend '{backend}'. The ONNX backend requires "
                    f"an .onnx model artifact.",
                    details={
                        "detected_framework": fw,
                        "architecture": arch,
                        "model_type": mtype,
                        "triton_backend": backend,
                        "model_name": model_name,
                        "resolved_path": local_path,
                    }
                )

        return fw, arch, mtype

    def stage_for_triton(
        self,
        local_path: str,
        triton_config: Optional[TritonConfig] = None,
        model_name: Optional[str] = None,
    ) -> str:
        """
        Stages a verified model artifact to the directory accessible by NVIDIA Triton.
        Ensures the artifact is available at the path expected by Triton's model configuration.

        Args:
            local_path: Path to the downloaded/local model file or directory.
            triton_config: Triton configuration with staging_dir or model_repository_path.
            model_name: Optional model identifier.

        Returns:
            Absolute path to the staged model artifact.
        """
        cfg = triton_config or TritonConfig()
        staging_dir = cfg.staging_dir

        if not staging_dir:
            if cfg.model_repository_path and model_name:
                staging_dir = os.path.join(cfg.model_repository_path, model_name, "1")
            else:
                return os.path.abspath(local_path)

        os.makedirs(staging_dir, exist_ok=True)

        if os.path.isfile(local_path):
            artifact_name = os.path.basename(local_path)
            dest_file = os.path.join(staging_dir, artifact_name)

            if os.path.exists(dest_file):
                try:
                    if os.path.samefile(local_path, dest_file):
                        return dest_file
                    if os.path.getsize(local_path) == os.path.getsize(dest_file):
                        logger.info(f"Model artifact already staged at '{dest_file}'. Reusing existing file.")
                        return dest_file
                except OSError:
                    pass

            shutil.copy2(local_path, dest_file)
            logger.info(f"Staged model artifact from '{local_path}' to '{dest_file}'.")
            return dest_file

        elif os.path.isdir(local_path):
            dir_name = model_name or os.path.basename(local_path.rstrip("/"))
            dest_dir = os.path.join(staging_dir, dir_name)

            if os.path.exists(dest_dir):
                try:
                    if os.path.samefile(local_path, dest_dir):
                        return dest_dir
                except OSError:
                    pass

            shutil.copytree(local_path, dest_dir, dirs_exist_ok=True)
            logger.info(f"Staged model directory from '{local_path}' to '{dest_dir}'.")
            return dest_dir

        return os.path.abspath(local_path)

    def retrieve_model_artifact(
        self,
        model_source: Optional[str] = None,
        config: Optional[FrameworkConfig] = None,
    ) -> str:
        """
        Retrieves a model artifact from MinIO / S3 storage or resolves a local path.
        Returns the resolved local directory or file path.
        """
        cfg = config or FrameworkConfig()
        source_str = (
            model_source
            or cfg.model_source
            or (f"s3://{cfg.model_bucket.strip()}/{cfg.model_key.lstrip('/')}" if cfg.model_bucket and cfg.model_key else None)
            or (f"s3://{cfg.model_bucket.strip()}/{cfg.model_dir.strip().strip('/')}" if cfg.model_bucket and cfg.model_dir else None)
        )
        if not source_str or not str(source_str).strip():
            raise ModelLoadError(
                "Model source cannot be empty or None. Provide model_source or (model_bucket and model_key/model_dir)."
            )

        source_str = str(source_str).strip()
        parsed = urlparse(source_str)
        scheme = parsed.scheme.lower()
        if scheme in self.SUPPORTED_STORAGE_SCHEMES:
            return self._download_remote_artifact(source_str)
        return self._clean_path(source_str)

    def load_for_triton(
        self,
        model_source: Optional[str] = None,
        config: Optional[FrameworkConfig] = None,
        inference_engine: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves a model artifact from MinIO / S3 storage (or local path),
        detects its format, validates compatibility with Triton's configured backend,
        stages the artifact for Triton, and returns a model metadata descriptor.

        Does NOT instantiate local model adapters or load model weights on the client host.
        Production inference is delegated exclusively to NVIDIA Triton.

        Args:
            model_source: Remote storage URI (s3://, minio://) or local path.
            config: Framework configuration containing storage and Triton options.
            inference_engine: Optional reference to the active TritonInferenceEngine.

        Returns:
            Dictionary containing model metadata, architecture, format, and staged path.

        Raises:
            ModelLoadError: If model_source is empty or resolution fails.
            ModelFormatError: If model format is incompatible with Triton backend.
            StorageError: If remote storage retrieval, authentication, or connectivity fails.
        """
        cfg = config or FrameworkConfig()
        source_str = (
            model_source
            or cfg.model_source
            or (f"s3://{cfg.model_bucket.strip()}/{cfg.model_key.lstrip('/')}" if cfg.model_bucket and cfg.model_key else None)
            or (f"s3://{cfg.model_bucket.strip()}/{cfg.model_dir.strip().strip('/')}" if cfg.model_bucket and cfg.model_dir else None)
        )
        is_remote = False
        if source_str:
            parsed = urlparse(str(source_str).strip())
            is_remote = parsed.scheme.lower() in self.SUPPORTED_STORAGE_SCHEMES

        local_path = self.retrieve_model_artifact(model_source, cfg)

        try:
            if not os.path.exists(local_path):
                raise ModelLoadError(
                    f"Resolved model path does not exist: '{local_path}'",
                    details={"model_source": source_str, "resolved_path": local_path}
                )

            triton_cfg = cfg.triton_config
            backend = triton_cfg.backend or "python"
            target_model_name = triton_cfg.model_name or cfg.model_name

            fw, arch, mtype = self.validate_backend_compatibility(
                local_path=local_path,
                backend=backend,
                model_name=target_model_name,
                configured_framework=cfg.framework,
            )

            staged_path = self.stage_for_triton(
                local_path=local_path,
                triton_config=triton_cfg,
                model_name=target_model_name,
            )

            task = self.format_detector.detect_task(
                model_path=local_path,
                architecture=arch,
                model_type=mtype,
                configured_task=cfg.task,
            )

            resolved_name = target_model_name or (
                os.path.splitext(os.path.basename(local_path))[0]
                if os.path.isfile(local_path)
                else os.path.basename(local_path.rstrip("/"))
            )

            metadata: Dict[str, Any] = {
                "model_name": resolved_name,
                "framework": fw,
                "architecture": arch,
                "model_type": mtype,
                "task": task,
                "input_type": cfg.input_type,
                "model_source": source_str,
                "local_path": local_path,
                "staged_path": staged_path,
                "triton_backend": backend,
                "is_triton_served": True,
            }
            logger.info(
                f"Model successfully loaded and staged for Triton: name='{resolved_name}', "
                f"framework='{fw}', architecture='{arch}', task='{task}', backend='{backend}', staged_path='{staged_path}'"
            )
            return metadata

        except Exception:
            if is_remote:
                self._cleanup_temp_dirs()
            raise

    def unload(self, model_instance: Any) -> None:
        """
        Safely unloads model instance, frees execution memory, and cleans up temporary downloads.
        """
        if model_instance is not None:
            if hasattr(model_instance, "unload") and callable(model_instance.unload):
                try:
                    model_instance.unload()
                except Exception as e:
                    logger.warning(f"Error during model instance unload: {e}")
                    raise

        self._cleanup_temp_dirs()

    def _cleanup_temp_dirs(self) -> None:
        """Removes temporary download directories created for remote artifacts."""
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
