"""
Configuration interfaces for the ML Framework.

Provides typed, validated configuration classes required for framework initialization
and lifecycle orchestration.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List, Set
from ..errors.exceptions import ConfigurationError


def parse_env_file(env_path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """
    Parses key-value pairs from a .env file without mutating os.environ.
    Finds .env in the specified path or walks up parent directories.
    """
    env_vars: Dict[str, str] = {}
    if env_path is not None:
        target = Path(env_path)
        candidates = [target] if target.is_file() else []
    else:
        candidates = []
        current = Path(__file__).resolve().parent
        for _ in range(6):
            env_candidate = current / ".env"
            if env_candidate.is_file():
                candidates.append(env_candidate)
                break
            if current.parent == current:
                break
            current = current.parent

    for candidate in candidates:
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("export "):
                        line = line[7:].strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k = k.strip()
                    v = v.strip()
                    if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                        v = v[1:-1]
                    elif " #" in v:
                        v = v.split(" #", 1)[0].rstrip()
                    if k and k not in env_vars:
                        env_vars[k] = v
        except Exception:
            pass
        break
    return env_vars


def load_env_file(env_path: Optional[Union[str, Path]] = None, override: bool = False) -> Dict[str, str]:
    """
    Loads key-value pairs from a .env file into os.environ.
    Preserves existing os.environ keys unless override=True.
    """
    parsed = parse_env_file(env_path)
    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return parsed


ALIAS_GROUPS: List[Set[str]] = [
    {"kafka_topic", "KAFKA_SOURCE_TOPIC", "KAFKA_TEST_TOPIC", "KAFKA_TOPIC"},
    {"kafka_bootstrap_servers", "KAFKA_BOOTSTRAP_SERVER", "KAFKA_BOOTSTRAP_SERVERS"},
    {"kafka_group_id", "KAFKA_GROUP_ID", "KAFKA_CONSUMER_GROUP"},
    {"endpoint_url", "MINIO_ENDPOINT", "SEAWEEDFS_ENDPOINT", "SEAWEEDFS_S3_ENDPOINT", "S3_ENDPOINT_URL"},
    {"access_key", "MINIO_ACCESS_KEY", "SEAWEEDFS_ACCESS_KEY", "AWS_ACCESS_KEY_ID"},
    {"secret_key", "MINIO_SECRET_KEY", "SEAWEEDFS_SECRET_KEY", "AWS_SECRET_ACCESS_KEY"},
    {"region", "MINIO_REGION", "AWS_DEFAULT_REGION"},
    {"model_bucket", "MINIO_MODEL_BUCKET", "S3_MODEL_BUCKET", "MINIO_TEST_BUCKET", "S3_TEST_BUCKET", "MINIO_BUCKET", "S3_BUCKET"},
    {"model_key", "MINIO_MODEL_KEY", "S3_MODEL_KEY", "model_dir", "MODEL_DIR", "object_key", "OBJECT_KEY"},
    {"verify_ssl", "MINIO_VERIFY_SSL", "S3_VERIFY_SSL", "MINIO_CA_BUNDLE", "AWS_CA_BUNDLE", "S3_CA_BUNDLE", "CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"},
    {"endpoint_url", "TRITON_SERVER_URL", "TRITON_URL", "TRITON_ENDPOINT"},
    {"model_name", "TRITON_MODEL_NAME"},
    {"task", "TASK", "MODEL_TASK"},
    {"input_type", "INPUT_TYPE"},
    {"device", "DEVICE", "ACCELERATOR"},
]


def _merge_env_layers(*layers: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges configuration layers in increasing order of precedence.
    Later layers override earlier layers.
    When a later layer defines a key belonging to an alias group,
    all keys from that alias group in earlier layers are purged
    so higher-precedence values take effect across all known synonyms.
    """
    merged: Dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        for group in ALIAS_GROUPS:
            if any(k in layer for k in group):
                for k in group:
                    merged.pop(k, None)
        merged.update(layer)
    return merged


@dataclass
class StorageConfig:
    """Configuration for artifact storage retrieval (e.g., S3 / MinIO / SeaweedFS via Boto3)."""
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = field(default=None, repr=False)
    region: str = "us-east-1"
    secure: bool = False
    download_dir: Optional[str] = None
    verify_ssl: Union[bool, str] = True
    model_bucket: Optional[str] = None
    model_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.endpoint_url and not self.endpoint_url.startswith(("http://", "https://")):
            self.endpoint_url = f"https://{self.endpoint_url}" if self.secure else f"http://{self.endpoint_url}"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "StorageConfig":
        """
        Constructs a StorageConfig strictly from the provided dictionary.
        Does not read os.environ or mutate environment state.
        """
        if not data:
            return cls()

        # Determine SSL certificate verification setting (default: True)
        verify_ssl: Union[bool, str] = True
        raw_verify = (
            data.get("verify_ssl")
            if data.get("verify_ssl") is not None
            else data.get("MINIO_VERIFY_SSL")
            if data.get("MINIO_VERIFY_SSL") is not None
            else data.get("S3_VERIFY_SSL")
        )
        if raw_verify is not None:
            if isinstance(raw_verify, bool):
                verify_ssl = raw_verify
            elif isinstance(raw_verify, str):
                val_lower = raw_verify.strip().lower()
                if val_lower in ("false", "0", "no"):
                    verify_ssl = False
                elif val_lower in ("true", "1", "yes"):
                    verify_ssl = True
                else:
                    verify_ssl = raw_verify.strip()  # path to custom CA certificate bundle
        elif (
            data.get("MINIO_CA_BUNDLE") is not None
            or data.get("AWS_CA_BUNDLE") is not None
            or data.get("S3_CA_BUNDLE") is not None
            or data.get("CA_BUNDLE") is not None
            or data.get("SSL_CERT_FILE") is not None
            or data.get("REQUESTS_CA_BUNDLE") is not None
            or data.get("CURL_CA_BUNDLE") is not None
        ):
            raw_ca = (
                data.get("MINIO_CA_BUNDLE")
                or data.get("AWS_CA_BUNDLE")
                or data.get("S3_CA_BUNDLE")
                or data.get("CA_BUNDLE")
                or data.get("SSL_CERT_FILE")
                or data.get("REQUESTS_CA_BUNDLE")
                or data.get("CURL_CA_BUNDLE")
            )
            if raw_ca and str(raw_ca).strip():
                verify_ssl = str(raw_ca).strip()
        elif (
            data.get("MLFLOW_S3_IGNORE_TLS") is not None
            or data.get("MINIO_IGNORE_TLS") is not None
            or data.get("S3_IGNORE_TLS") is not None
        ):
            raw_ignore = (
                data.get("MLFLOW_S3_IGNORE_TLS")
                if data.get("MLFLOW_S3_IGNORE_TLS") is not None
                else data.get("MINIO_IGNORE_TLS")
                if data.get("MINIO_IGNORE_TLS") is not None
                else data.get("S3_IGNORE_TLS")
            )
            ignore_tls = (
                raw_ignore
                if isinstance(raw_ignore, bool)
                else str(raw_ignore).strip().lower() in ("true", "1", "yes")
            )
            verify_ssl = not ignore_tls

        endpoint = (
            data.get("endpoint_url")
            or data.get("MINIO_ENDPOINT")
            or data.get("SEAWEEDFS_ENDPOINT")
            or data.get("SEAWEEDFS_S3_ENDPOINT")
            or data.get("S3_ENDPOINT_URL")
        )
        access = (
            data.get("access_key")
            or data.get("MINIO_ACCESS_KEY")
            or data.get("SEAWEEDFS_ACCESS_KEY")
            or data.get("AWS_ACCESS_KEY_ID")
        )
        secret = (
            data.get("secret_key")
            or data.get("MINIO_SECRET_KEY")
            or data.get("SEAWEEDFS_SECRET_KEY")
            or data.get("AWS_SECRET_ACCESS_KEY")
        )
        reg = (
            data.get("region")
            or data.get("MINIO_REGION")
            or data.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        sec_val = (
            data.get("secure")
            if data.get("secure") is not None
            else data.get("MINIO_SECURE")
            if data.get("MINIO_SECURE") is not None
            else data.get("SEAWEEDFS_SECURE")
            if data.get("SEAWEEDFS_SECURE") is not None
            else False
        )
        secure_flag = (
            sec_val
            if isinstance(sec_val, bool)
            else str(sec_val).strip().lower() in ("true", "1", "yes")
        )

        dl_dir = (
            data.get("download_dir")
            or data.get("MINIO_DOWNLOAD_DIR")
        )
        model_bkt = (
            data.get("model_bucket")
            or data.get("MINIO_MODEL_BUCKET")
            or data.get("S3_MODEL_BUCKET")
            or data.get("MINIO_BUCKET")
            or data.get("S3_BUCKET")
        )
        model_k = (
            data.get("model_key")
            or data.get("MINIO_MODEL_KEY")
            or data.get("S3_MODEL_KEY")
            or data.get("model_dir")
            or data.get("MODEL_DIR")
            or data.get("object_key")
            or data.get("OBJECT_KEY")
        )

        cfg = cls(
            endpoint_url=endpoint,
            access_key=access,
            secret_key=secret,
            region=reg,
            secure=secure_flag,
            download_dir=dl_dir,
            verify_ssl=verify_ssl,
            model_bucket=model_bkt,
            model_key=model_k,
        )
        return cfg

    @classmethod
    def from_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        load_dotenv: bool = True,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> "StorageConfig":
        """
        Builds a StorageConfig by resolving values in order of precedence:
        1. Explicitly provided env dict
        2. os.environ
        3. .env file (if load_dotenv is True)
        Does not mutate global os.environ.
        """
        merged = _merge_env_layers(
            parse_env_file(dotenv_path) if load_dotenv else None,
            dict(os.environ),
            env,
        )
        return cls.from_dict(merged)

    def validate_live_config(self, bucket: Optional[str] = None) -> None:
        """
        Validates that required live MinIO / S3 configuration parameters are present
        and contain actionable, non-placeholder values.
        """
        target_bucket = bucket if bucket is not None else self.model_bucket
        missing = []
        if not self.endpoint_url or not self.endpoint_url.strip():
            missing.append("MINIO_ENDPOINT (or S3_ENDPOINT_URL)")
        if not self.access_key or not self.access_key.strip():
            missing.append("MINIO_ACCESS_KEY (or AWS_ACCESS_KEY_ID)")
        if not self.secret_key or not self.secret_key.strip():
            missing.append("MINIO_SECRET_KEY (or AWS_SECRET_ACCESS_KEY)")
        if target_bucket is not None and not target_bucket.strip():
            missing.append("MINIO_TEST_BUCKET (or MINIO_MODEL_BUCKET / S3_TEST_BUCKET)")

        if missing:
            raise ConfigurationError(
                f"Missing required live storage configuration: {', '.join(missing)}. "
                f"Please define them via environment variables or configuration.",
                details={"missing": missing}
            )

        # Check for unconfigured placeholders
        placeholders = {
            "your_minio_access_key_here",
            "your_minio_secret_key_here",
            "your_key_here",
            "<minio-access-key>",
            "<minio-secret-key>",
        }
        if (
            self.access_key in placeholders
            or self.secret_key in placeholders
            or (self.access_key and "<" in self.access_key)
            or (self.secret_key and "<" in self.secret_key)
            or (self.endpoint_url and "<" in self.endpoint_url)
        ):
            raise ConfigurationError(
                "Placeholder credentials detected in storage configuration. "
                "Please provide your team's authorized MinIO credentials via environment variables.",
                details={"access_key_placeholder": True}
            )

        if isinstance(self.verify_ssl, str):
            ca_path = self.verify_ssl.strip()
            if not ca_path:
                raise ConfigurationError("Configured CA bundle path cannot be empty.")
            if not os.path.isfile(ca_path) and not ca_path.startswith("<"):
                raise ConfigurationError(
                    f"Configured CA bundle file does not exist: '{ca_path}'",
                    details={"verify_ssl": ca_path}
                )


@dataclass
class InputConfig:
    """Configuration for input source ingestion, batching, and chunking."""
    source: Optional[Any] = None
    source_type: str = "auto"  # "auto", "file", "object_storage", "kafka", "direct"
    batch_size: int = 1
    chunk_size: Optional[int] = None  # e.g., byte chunk size or line count for streaming
    encoding: str = "utf-8"
    kafka_topic: Optional[str] = None
    kafka_bootstrap_servers: Optional[str] = None
    kafka_group_id: Optional[str] = None
    kafka_options: Dict[str, Any] = field(default_factory=dict)
    storage_config: Optional[StorageConfig] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validates input configuration values."""
        if self.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be >= 1, got {self.batch_size}",
                details={"batch_size": self.batch_size}
            )
        if self.chunk_size is not None and self.chunk_size <= 0:
            raise ConfigurationError(
                f"chunk_size must be > 0, got {self.chunk_size}",
                details={"chunk_size": self.chunk_size}
            )
        valid_source_types = {"auto", "file", "object_storage", "kafka", "direct"}
        if self.source_type not in valid_source_types:
            raise ConfigurationError(
                f"Invalid source_type '{self.source_type}'. Allowed types: {valid_source_types}",
                details={"source_type": self.source_type, "allowed": list(valid_source_types)}
            )

    def validate_live_kafka_config(self) -> None:
        """
        Validates that required live Kafka streaming configuration parameters are present
        and contain actionable, non-placeholder values.
        """
        missing = []
        if not self.kafka_bootstrap_servers or not self.kafka_bootstrap_servers.strip():
            missing.append("KAFKA_BOOTSTRAP_SERVER")
        if not self.kafka_topic or not self.kafka_topic.strip():
            missing.append("KAFKA_TEST_TOPIC (or KAFKA_SOURCE_TOPIC)")

        if missing:
            raise ConfigurationError(
                f"Missing required live Kafka configuration: {', '.join(missing)}. "
                f"Please define them via environment variables or configuration.",
                details={"missing": missing}
            )

        if (
            self.kafka_bootstrap_servers == "<kafka-host>:<port>"
            or "<" in self.kafka_bootstrap_servers
        ):
            raise ConfigurationError(
                "Placeholder Kafka bootstrap server detected in configuration. "
                "Please configure KAFKA_BOOTSTRAP_SERVER with your team's running Kafka broker.",
                details={"kafka_bootstrap_servers": self.kafka_bootstrap_servers}
            )

        import re
        if not re.match(r"^[a-zA-Z0-9._-]+$", self.kafka_topic):
            raise ConfigurationError(
                f"Invalid Kafka topic name '{self.kafka_topic}'. Topic must contain only alphanumeric, '.', '_', or '-'"
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputConfig":
        if not data:
            return cls()
        storage_data = data.get("storage_config")
        storage_cfg = (
            storage_data
            if isinstance(storage_data, StorageConfig)
            else StorageConfig.from_dict(storage_data)
            if isinstance(storage_data, dict)
            else None
        )
        kafka_opts = dict(data.get("kafka_options") or {})
        if "security.protocol" not in kafka_opts and data.get("KAFKA_SECURITY_PROTOCOL"):
            kafka_opts["security.protocol"] = data.get("KAFKA_SECURITY_PROTOCOL")
        if "sasl.mechanism" not in kafka_opts and data.get("KAFKA_SASL_MECHANISM"):
            kafka_opts["sasl.mechanism"] = data.get("KAFKA_SASL_MECHANISM")
        if "sasl.username" not in kafka_opts and data.get("KAFKA_SASL_USERNAME"):
            kafka_opts["sasl.username"] = data.get("KAFKA_SASL_USERNAME")
        if "sasl.password" not in kafka_opts and data.get("KAFKA_SASL_PASSWORD"):
            kafka_opts["sasl.password"] = data.get("KAFKA_SASL_PASSWORD")
        if "ssl.ca.location" not in kafka_opts and data.get("KAFKA_SSL_CA_LOCATION"):
            kafka_opts["ssl.ca.location"] = data.get("KAFKA_SSL_CA_LOCATION")
        if "ssl.certificate.location" not in kafka_opts and data.get("KAFKA_SSL_CERTIFICATE_LOCATION"):
            kafka_opts["ssl.certificate.location"] = data.get("KAFKA_SSL_CERTIFICATE_LOCATION")
        if "ssl.key.location" not in kafka_opts and data.get("KAFKA_SSL_KEY_LOCATION"):
            kafka_opts["ssl.key.location"] = data.get("KAFKA_SSL_KEY_LOCATION")

        cfg = cls(
            source=data.get("source") or data.get("input_source"),
            source_type=data.get("source_type", "auto"),
            batch_size=int(data.get("batch_size", 1)),
            chunk_size=int(data.get("chunk_size")) if data.get("chunk_size") is not None else None,
            encoding=data.get("encoding", "utf-8"),
            kafka_topic=data.get("kafka_topic") or data.get("KAFKA_SOURCE_TOPIC") or data.get("KAFKA_TEST_TOPIC") or data.get("KAFKA_TOPIC"),
            kafka_bootstrap_servers=data.get("kafka_bootstrap_servers") or data.get("KAFKA_BOOTSTRAP_SERVER") or data.get("KAFKA_BOOTSTRAP_SERVERS"),
            kafka_group_id=data.get("kafka_group_id") or data.get("KAFKA_GROUP_ID") or data.get("KAFKA_CONSUMER_GROUP"),
            kafka_options=kafka_opts,
            storage_config=storage_cfg,
            custom_options=data.get("custom_options", {}),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        load_dotenv: bool = True,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> "InputConfig":
        """
        Builds an InputConfig by resolving values in order of precedence:
        1. Explicitly provided env dict
        2. os.environ
        3. .env file (if load_dotenv is True)
        Does not mutate global os.environ.
        """
        merged = _merge_env_layers(
            parse_env_file(dotenv_path) if load_dotenv else None,
            dict(os.environ),
            env,
        )
        return cls.from_dict(merged)


@dataclass
class TritonConfig:
    """Configuration for NVIDIA Triton Inference Server execution."""
    endpoint_url: str = "http://localhost:8010"
    protocol: str = "http"  # "http" or "grpc"
    model_name: Optional[str] = None
    model_version: str = ""
    backend: str = "python"
    staging_dir: Optional[str] = None
    model_repository_path: Optional[str] = None
    timeout_seconds: float = 30.0
    check_model_ready: bool = True
    max_retries: int = 3
    retry_interval_seconds: float = 0.5
    input_tensor_map: Dict[str, Any] = field(default_factory=dict)
    output_tensor_map: Dict[str, Any] = field(default_factory=dict)
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.endpoint_url and self.protocol == "http" and not self.endpoint_url.startswith(("http://", "https://")):
            self.endpoint_url = f"http://{self.endpoint_url}"

    def validate(self) -> None:
        """Validates Triton configuration parameters."""
        if not self.endpoint_url or not str(self.endpoint_url).strip():
            raise ConfigurationError(
                "Triton endpoint_url cannot be empty.",
                details={"endpoint_url": self.endpoint_url}
            )
        valid_protocols = {"http", "grpc"}
        if self.protocol.lower().strip() not in valid_protocols:
            raise ConfigurationError(
                f"Invalid Triton protocol '{self.protocol}'. Supported protocols: {valid_protocols}",
                details={"protocol": self.protocol, "allowed": list(valid_protocols)}
            )
        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                f"Triton timeout_seconds must be > 0, got {self.timeout_seconds}",
                details={"timeout_seconds": self.timeout_seconds}
            )
        if self.max_retries < 0:
            raise ConfigurationError(
                f"Triton max_retries must be >= 0, got {self.max_retries}",
                details={"max_retries": self.max_retries}
            )
        if self.retry_interval_seconds < 0:
            raise ConfigurationError(
                f"Triton retry_interval_seconds must be >= 0, got {self.retry_interval_seconds}",
                details={"retry_interval_seconds": self.retry_interval_seconds}
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TritonConfig":
        if not data:
            return cls()
        endpoint = (
            data.get("endpoint_url")
            or data.get("TRITON_SERVER_URL")
            or data.get("TRITON_HTTP_ENDPOINT")
            or data.get("TRITON_URL")
            or data.get("TRITON_ENDPOINT")
            or "http://localhost:8010"
        )
        proto = data.get("protocol") or data.get("TRITON_PROTOCOL") or "http"
        name = data.get("model_name") or data.get("TRITON_MODEL_NAME")
        ver = str(data.get("model_version") or data.get("TRITON_MODEL_VERSION") or "")
        timeout = float(
            data.get("timeout_seconds")
            or data.get("TRITON_TIMEOUT")
            or data.get("TRITON_TIMEOUT_SECONDS")
            or 30.0
        )
        check_ready_val = (
            data.get("check_model_ready")
            if data.get("check_model_ready") is not None
            else data.get("TRITON_CHECK_READY")
        )
        if check_ready_val is None:
            check_ready = True
        elif isinstance(check_ready_val, bool):
            check_ready = check_ready_val
        else:
            check_ready = str(check_ready_val).strip().lower() in ("true", "1", "yes")

        max_retries = int(data.get("max_retries") or data.get("TRITON_MAX_RETRIES") or 3)
        retry_interval = float(data.get("retry_interval_seconds") or data.get("TRITON_RETRY_INTERVAL") or 0.5)

        backend = str(data.get("backend") or data.get("TRITON_BACKEND") or "python")
        staging_dir = (
            data.get("staging_dir")
            or data.get("TRITON_STAGING_DIR")
            or data.get("TRITON_MODEL_STAGING_DIR")
        )
        if not staging_dir and os.path.isdir("/tmp/poc_nemo_model"):
            staging_dir = "/tmp/poc_nemo_model"

        model_repo = (
            data.get("model_repository_path")
            or data.get("TRITON_MODEL_REPO")
            or data.get("TRITON_MODEL_REPOSITORY")
        )
        if not model_repo and os.path.isdir("/tmp/poc_triton_models_python"):
            model_repo = "/tmp/poc_triton_models_python"

        cfg = cls(
            endpoint_url=str(endpoint),
            protocol=str(proto),
            model_name=str(name) if name is not None else None,
            model_version=ver,
            backend=backend,
            staging_dir=staging_dir,
            model_repository_path=model_repo,
            timeout_seconds=timeout,
            check_model_ready=check_ready,
            max_retries=max_retries,
            retry_interval_seconds=retry_interval,
            input_tensor_map=dict(data.get("input_tensor_map") or {}),
            output_tensor_map=dict(data.get("output_tensor_map") or {}),
            custom_options=dict(data.get("custom_options") or {}),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        load_dotenv: bool = True,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> "TritonConfig":
        merged: Dict[str, Any] = {}
        if load_dotenv:
            merged.update(parse_env_file(dotenv_path))
        merged.update(dict(os.environ))
        if env:
            merged.update(env)
        return cls.from_dict(merged)


@dataclass
class FrameworkConfig:
    """
    Primary configuration container for the ExecutionManager.
    Specifies model resolution, hardware execution preferences, stage options, and input sources.
    """
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_source: Optional[str] = None
    model_bucket: Optional[str] = None
    model_key: Optional[str] = None
    model_dir: Optional[str] = None
    model_id: Optional[str] = None
    task: str = "auto"
    input_type: str = "auto"
    framework: str = "auto"
    deployment_point: Optional[str] = None
    device: str = "auto"
    batch_size: int = 1
    timeout_seconds: float = 30.0
    enable_telemetry: bool = True
    auto_cleanup: bool = True
    storage_config: StorageConfig = field(default_factory=StorageConfig)
    input_config: InputConfig = field(default_factory=InputConfig)
    triton_config: TritonConfig = field(default_factory=TritonConfig)
    input_source: Optional[Any] = None
    custom_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_key and self.model_dir:
            self.model_key = self.model_dir
        if not self.model_name and self.model_id:
            self.model_name = self.model_id

        if not self.model_source and self.model_bucket and self.model_key:
            self.model_source = f"s3://{str(self.model_bucket).strip()}/{str(self.model_key).lstrip('/')}"
        elif self.model_source and (not self.model_bucket or not self.model_key):
            if str(self.model_source).startswith(("s3://", "minio://", "seaweedfs://")):
                from urllib.parse import urlparse
                p = urlparse(str(self.model_source))
                if not self.model_bucket:
                    self.model_bucket = p.netloc
                if not self.model_key:
                    self.model_key = p.path.lstrip("/")

    def validate(self) -> None:
        """Validates configuration constraints, raising ConfigurationError on invalid values."""
        if self.batch_size < 1:
            raise ConfigurationError(
                f"batch_size must be >= 1, got {self.batch_size}",
                details={"batch_size": self.batch_size}
            )

        if self.timeout_seconds <= 0:
            raise ConfigurationError(
                f"timeout_seconds must be > 0, got {self.timeout_seconds}",
                details={"timeout_seconds": self.timeout_seconds}
            )

        valid_devices = {"auto", "cpu", "cuda", "mps"}
        device_lower = self.device.lower().strip()
        is_cuda_idx = device_lower.startswith("cuda:") and device_lower[5:].isdigit()
        if device_lower not in valid_devices and not is_cuda_idx:
            raise ConfigurationError(
                f"Invalid device '{self.device}'. Must be one of {valid_devices} or 'cuda:<index>'",
                details={"device": self.device, "allowed": list(valid_devices)}
            )

        # Validate task
        valid_tasks = {"auto", "transcription", "language_detection", "denoising"}
        norm_task = (self.task or "auto").lower().strip()
        if norm_task not in valid_tasks:
            raise ConfigurationError(
                f"Invalid task '{self.task}'. Supported tasks: {sorted(list(valid_tasks))}",
                details={"task": self.task, "allowed": sorted(list(valid_tasks))}
            )

        # Validate input_type
        valid_input_types = {"auto", "audio", "text"}
        norm_input_type = (self.input_type or "auto").lower().strip()
        if norm_input_type not in valid_input_types:
            raise ConfigurationError(
                f"Invalid input_type '{self.input_type}'. Supported input types: {sorted(list(valid_input_types))}",
                details={"input_type": self.input_type, "allowed": sorted(list(valid_input_types))}
            )

        # Cross-validation between task and input_type
        if norm_task == "denoising" and norm_input_type == "text":
            raise ConfigurationError(
                "Task 'denoising' is incompatible with input_type 'text'. Denoising requires audio input.",
                details={"task": self.task, "input_type": self.input_type}
            )
        if norm_task == "transcription" and norm_input_type == "text":
            raise ConfigurationError(
                "Task 'transcription' is incompatible with input_type 'text'. Transcription requires audio input.",
                details={"task": self.task, "input_type": self.input_type}
            )

        if self.input_config:
            self.input_config.validate()

        if self.triton_config:
            self.triton_config.validate()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameworkConfig":
        if not isinstance(data, dict):
            raise ConfigurationError(f"Expected dict for FrameworkConfig, got {type(data).__name__}")

        storage_dict = data.get("storage_config")
        storage_cfg = (
            storage_dict
            if isinstance(storage_dict, StorageConfig)
            else StorageConfig.from_dict(storage_dict if isinstance(storage_dict, dict) and storage_dict else data)
        )

        input_dict = data.get("input_config")
        if isinstance(input_dict, InputConfig):
            input_cfg = input_dict
        elif isinstance(input_dict, dict) and input_dict:
            input_dict_copy = dict(input_dict)
            if data.get("input_source") and "source" not in input_dict_copy:
                input_dict_copy["source"] = data.get("input_source")
            input_cfg = InputConfig.from_dict(input_dict_copy)
        else:
            flat_input = dict(data)
            if data.get("input_source") and "source" not in flat_input:
                flat_input["source"] = data.get("input_source")
            input_cfg = InputConfig.from_dict(flat_input)

        triton_dict = data.get("triton_config")
        triton_cfg = (
            triton_dict
            if isinstance(triton_dict, TritonConfig)
            else TritonConfig.from_dict(triton_dict if isinstance(triton_dict, dict) and triton_dict else data)
        )

        m_name = (
            data.get("model_name")
            or data.get("model_id")
            or data.get("MODEL_ID")
            or data.get("model_identifier")
            or data.get("MODEL_IDENTIFIER")
        )
        if triton_cfg.model_name is None and m_name:
            triton_cfg.model_name = m_name

        bucket = (
            data.get("model_bucket")
            or data.get("MINIO_MODEL_BUCKET")
            or data.get("S3_MODEL_BUCKET")
            or data.get("MINIO_BUCKET")
            or data.get("S3_BUCKET")
            or (storage_cfg.model_bucket if storage_cfg else None)
        )
        key = (
            data.get("model_key")
            or data.get("MINIO_MODEL_KEY")
            or data.get("S3_MODEL_KEY")
            or data.get("model_dir")
            or data.get("MODEL_DIR")
            or data.get("object_key")
            or data.get("OBJECT_KEY")
            or (storage_cfg.model_key if storage_cfg else None)
        )
        m_dir = data.get("model_dir") or data.get("MODEL_DIR")
        m_id = data.get("model_id") or data.get("MODEL_ID") or data.get("model_identifier") or m_name

        model_src = data.get("model_source")
        if not model_src and bucket and key:
            model_src = f"s3://{str(bucket).strip()}/{str(key).lstrip('/')}"
        elif model_src and (not bucket or not key):
            if str(model_src).startswith(("s3://", "minio://", "seaweedfs://")):
                from urllib.parse import urlparse
                p = urlparse(str(model_src))
                if not bucket:
                    bucket = p.netloc
                if not key:
                    key = p.path.lstrip("/")

        task = str(data.get("task") or data.get("TASK") or data.get("MODEL_TASK") or "auto")
        input_type = str(data.get("input_type") or data.get("INPUT_TYPE") or "auto")

        config = cls(
            model_name=m_name,
            model_version=str(data.get("model_version")) if data.get("model_version") is not None else None,
            model_source=model_src,
            model_bucket=bucket,
            model_key=key,
            model_dir=m_dir,
            model_id=m_id,
            task=task,
            input_type=input_type,
            framework=data.get("framework", "auto"),
            deployment_point=data.get("deployment_point"),
            device=data.get("device", "auto"),
            batch_size=int(data.get("batch_size", 1)),
            timeout_seconds=float(data.get("timeout_seconds", 30.0)),
            enable_telemetry=bool(data.get("enable_telemetry", True)),
            auto_cleanup=bool(data.get("auto_cleanup", True)),
            storage_config=storage_cfg,
            input_config=input_cfg,
            triton_config=triton_cfg,
            input_source=data.get("input_source") or input_cfg.source,
            custom_options=data.get("custom_options", {}),
        )
        config.validate()
        return config

    @classmethod
    def from_env(
        cls,
        env: Optional[Dict[str, str]] = None,
        load_dotenv: bool = True,
        dotenv_path: Optional[Union[str, Path]] = None,
    ) -> "FrameworkConfig":
        """
        Builds a FrameworkConfig by resolving values in order of precedence:
        1. Explicitly provided env dict
        2. os.environ
        3. .env file (if load_dotenv is True)
        Does not mutate global os.environ.
        """
        merged = _merge_env_layers(
            parse_env_file(dotenv_path) if load_dotenv else None,
            dict(os.environ),
            env,
        )
        return cls.from_dict(merged)
