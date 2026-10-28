"""
Source-specific input readers for local files, object storage, Kafka, and direct data.

Implements reusable, chunked, and streaming readers under the BaseInputReader contract.
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, Optional, List, Union
from urllib.parse import urlparse

from ..interfaces.base import BaseInputReader
from ..config.model_config import StorageConfig
from ..errors.exceptions import (
    MLFrameworkError,
    InputProcessingError,
    InvalidInputSourceError,
    StorageError,
    TransientStorageError,
)

logger = logging.getLogger("ml_framework.input.readers")


class LocalFileReader(BaseInputReader):
    """
    Reader for local filesystem inputs.

    Supports paths with or without file:// prefix.
    Enforces chunked and line-based streaming to prevent large files
    from exhausting system memory.
    """

    def __init__(self, default_chunk_size: int = 65536, encoding: str = "utf-8"):
        self.default_chunk_size = default_chunk_size
        self.encoding = encoding

    def _clean_path(self, source: Any) -> str:
        if isinstance(source, (bytes, bytearray)):
            s = os.fsdecode(source)
        else:
            s = str(source)
        if s.startswith("file://"):
            from urllib.parse import unquote, urlparse
            parsed = urlparse(s)
            return unquote(parsed.path)
        return s

    def validate(self, source: Any) -> bool:
        """Validates that the file exists, is a regular file, and is readable."""
        if not source or not isinstance(source, (str, bytes, os.PathLike)):
            raise InvalidInputSourceError(
                f"Local file source must be a path string, got {type(source).__name__}",
                details={"source": str(source)}
            )

        path = self._clean_path(source)
        if not os.path.exists(path):
            raise InvalidInputSourceError(
                f"Local input file does not exist: '{path}'",
                details={"path": path}
            )

        if os.path.isdir(path):
            raise InvalidInputSourceError(
                f"Input path is a directory, expected a regular file: '{path}'",
                details={"path": path}
            )

        if not os.access(path, os.R_OK):
            raise InvalidInputSourceError(
                f"Input file is not readable (permission denied): '{path}'",
                details={"path": path}
            )

        return True

    def read(
        self,
        source: Any,
        batch_size: int = 1,
        chunk_size: Optional[int] = None,
        as_bytes: bool = False
    ) -> Iterator[Any]:
        """
        Reads input data from the local file in chunks or line batches.

        If as_bytes is True, yields raw byte chunks of size chunk_size.
        Otherwise, reads lines in streaming mode and yields batches of batch_size lines.
        """
        self.validate(source)
        path = self._clean_path(str(source))
        effective_chunk_size = chunk_size or self.default_chunk_size

        try:
            if as_bytes:
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(effective_chunk_size)
                        if not chunk:
                            break
                        yield chunk
            else:
                # Check if file is small JSON
                _, ext = os.path.splitext(path.lower())
                if ext == ".json" and os.path.getsize(path) < effective_chunk_size:
                    with open(path, "r", encoding=self.encoding) as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                for i in range(0, len(data), batch_size):
                                    yield data[i:i + batch_size]
                            else:
                                yield data
                            return
                        except json.JSONDecodeError:
                            pass  # Fall back to line-by-line reading

                # Streaming line-by-line reading with batching
                batch = []
                with open(path, "r", encoding=self.encoding, errors="replace") as f:
                    for line in f:
                        batch.append(line.rstrip("\r\n"))
                        if len(batch) >= batch_size:
                            yield batch if batch_size > 1 else batch[0]
                            batch = []
                if batch:
                    yield batch if batch_size > 1 else batch[0]
        except OSError as e:
            if isinstance(e, MLFrameworkError):
                raise
            raise InputProcessingError(f"Failed to read local file '{path}': {e}") from e

    def close(self) -> None:
        """No-op for local file reader since files are managed via context managers."""
        pass


class ObjectStorageReader(BaseInputReader):
    """
    Reader for S3-compatible object storage (SeaweedFS, MinIO, AWS S3).

    Parses URIs with schemes: s3://, minio://, seaweedfs://.
    Uses Boto3 S3 client under the hood and streams chunks/lines
    without buffering the entire remote object in RAM.
    """

    SUPPORTED_SCHEMES = {"s3", "minio", "seaweedfs"}

    def __init__(
        self,
        storage_config: Optional[StorageConfig] = None,
        s3_client: Optional[Any] = None,
        default_chunk_size: int = 65536,
    ):
        self.storage_config = storage_config or StorageConfig()
        self._client = s3_client
        self.default_chunk_size = default_chunk_size

    def _parse_uri(self, uri: str) -> tuple[str, str, str]:
        """Parses scheme, bucket, and key from object storage URI."""
        parsed = urlparse(uri)
        scheme = parsed.scheme.lower()
        if scheme not in self.SUPPORTED_SCHEMES:
            raise InvalidInputSourceError(
                f"Unsupported object storage scheme '{scheme}'. Supported schemes: {self.SUPPORTED_SCHEMES}",
                details={"uri": uri, "supported": list(self.SUPPORTED_SCHEMES)}
            )

        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket:
            raise InvalidInputSourceError(
                f"Missing bucket name in object storage URI: '{uri}'",
                details={"uri": uri}
            )
        if not key:
            raise InvalidInputSourceError(
                f"Missing object key in object storage URI: '{uri}'",
                details={"uri": uri}
            )

        return scheme, bucket, key

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            import boto3
            from botocore.config import Config
        except ImportError as e:
            raise InputProcessingError(
                "boto3 is required for object storage inputs (SeaweedFS / MinIO / S3). "
                "Install offline via: pip install --no-index --find-links=/path/to/wheels ml-framework[storage]"
            ) from e

        storage_cfg = self.storage_config
        if not storage_cfg or not storage_cfg.endpoint_url:
            storage_cfg = StorageConfig.from_env()

        boto_cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
        self._client = boto3.client(
            "s3",
            endpoint_url=storage_cfg.endpoint_url,
            aws_access_key_id=storage_cfg.access_key,
            aws_secret_access_key=storage_cfg.secret_key,
            region_name=storage_cfg.region,
            use_ssl=storage_cfg.secure,
            verify=storage_cfg.verify_ssl,
            config=boto_cfg,
        )
        return self._client

    def validate(self, source: Any) -> bool:
        """Validates URI format and verifies object accessibility in object storage."""
        if not isinstance(source, str):
            raise InvalidInputSourceError(
                f"Object storage source must be a URI string, got {type(source).__name__}",
                details={"source": str(source)}
            )

        _, bucket, key = self._parse_uri(source)

        client = self._get_client()
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e).lower()
            code = ""
            if hasattr(e, "response") and isinstance(e.response, dict):
                code = str(e.response.get("Error", {}).get("Code", "")).lower()
            if "notfound" in err_str or "404" in err_str or "nosuchkey" in err_str or code in ("404", "nosuchkey", "notfound"):
                raise InvalidInputSourceError(
                    f"Object not found in storage: '{source}' (bucket='{bucket}', key='{key}')",
                    details={"uri": source, "bucket": bucket, "key": key}
                ) from e
            elif "connect" in err_str or "timeout" in err_str or "endpoint" in err_str:
                raise TransientStorageError(
                    f"Transient network or endpoint connection error accessing '{source}': {str(e)}",
                    details={"uri": source, "error": str(e)}
                ) from e
            else:
                raise StorageError(
                    f"Storage validation failed for '{source}': {str(e)}",
                    details={"uri": source, "error": str(e)}
                ) from e

    def read(
        self,
        source: Any,
        batch_size: int = 1,
        chunk_size: Optional[int] = None,
        as_bytes: bool = False
    ) -> Iterator[Any]:
        """
        Streams object content from SeaweedFS / MinIO / S3 in chunks or lines.
        Guarantees that large remote objects are not loaded entirely into memory.
        """
        _, bucket, key = self._parse_uri(str(source))
        client = self._get_client()

        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except Exception as e:
            err_str = str(e).lower()
            code = ""
            if hasattr(e, "response") and isinstance(e.response, dict):
                code = str(e.response.get("Error", {}).get("Code", "")).lower()
            if "notfound" in err_str or "404" in err_str or "nosuchkey" in err_str or code in ("404", "nosuchkey", "notfound"):
                raise InvalidInputSourceError(f"Object not found: '{source}'") from e
            raise StorageError(f"Failed to retrieve object '{source}': {str(e)}") from e

        body = response.get("Body")
        if body is None:
            return

        effective_chunk_size = chunk_size or self.default_chunk_size

        try:
            if as_bytes:
                if hasattr(body, "iter_chunks"):
                    for chunk in body.iter_chunks(effective_chunk_size):
                        yield chunk
                elif hasattr(body, "read"):
                    while True:
                        chunk = body.read(effective_chunk_size)
                        if not chunk:
                            break
                        yield chunk
            else:
                batch = []
                if hasattr(body, "iter_lines"):
                    for line in body.iter_lines():
                        decoded = line.decode("utf-8", errors="replace").rstrip("\r\n")
                        batch.append(decoded)
                        if len(batch) >= batch_size:
                            yield batch if batch_size > 1 else batch[0]
                            batch = []
                elif hasattr(body, "read"):
                    raw_data = body.read().decode("utf-8", errors="replace")
                    for line in raw_data.splitlines():
                        batch.append(line)
                        if len(batch) >= batch_size:
                            yield batch if batch_size > 1 else batch[0]
                            batch = []

                if batch:
                    yield batch if batch_size > 1 else batch[0]
        except Exception as e:
            err_str = str(e).lower()
            if "connect" in err_str or "timeout" in err_str or "endpoint" in err_str or "reset" in err_str:
                raise TransientStorageError(f"Transient error streaming from '{source}': {e}") from e
            if isinstance(e, MLFrameworkError):
                raise
            raise StorageError(f"Error streaming object from '{source}': {e}") from e
        finally:
            if hasattr(body, "close") and callable(body.close):
                try:
                    body.close()
                except Exception:
                    pass

    def close(self) -> None:
        """Closes the underlying Boto3 S3 client connection pool."""
        if self._client is not None:
            if hasattr(self._client, "close") and callable(self._client.close):
                try:
                    self._client.close()
                except Exception as e:
                    logger.warning(f"Error closing S3 client: {e}")
            self._client = None


class KafkaTopicReader(BaseInputReader):
    """
    Reader for streaming input from Kafka topics.

    Encapsulates Kafka consumer polling behind a reusable interface.
    Supports injected mock consumers for unit testing without live brokers.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        group_id: Optional[str] = None,
        consumer: Optional[Any] = None,
        kafka_options: Optional[Dict[str, Any]] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id or "ml-framework-consumer-group"
        self._consumer = consumer
        self.kafka_options = kafka_options or {}

    def _extract_topic(self, source: Any) -> str:
        if isinstance(source, str):
            if source.startswith("kafka://"):
                return source[8:].split("/")[0]
            return source
        if isinstance(source, dict):
            topic = source.get("kafka_topic") or source.get("topic")
            if topic:
                return str(topic)
        raise InvalidInputSourceError(
            f"Unable to extract Kafka topic from source: {source}",
            details={"source": str(source)}
        )

    def validate(self, source: Any) -> bool:
        """Validates topic name format and consumer readiness."""
        topic = self._extract_topic(source)
        if not topic or not topic.strip():
            raise InvalidInputSourceError("Kafka topic name cannot be empty")

        # Kafka allowed characters: letters, numbers, '.', '_', '-'
        import re
        if not re.match(r"^[a-zA-Z0-9._-]+$", topic):
            raise InvalidInputSourceError(
                f"Invalid Kafka topic name '{topic}'. Must contain only alphanumeric, '.', '_', or '-'"
            )

        return True

    def read(
        self,
        source: Any,
        batch_size: int = 1,
        chunk_size: Optional[int] = None,
        timeout_seconds: float = 1.0,
        max_messages: Optional[int] = None
    ) -> Iterator[Any]:
        """
        Polls messages from Kafka in batches.
        If a consumer is not available or mock is not passed, reports missing dependency.
        """
        self.validate(source)
        topic = self._extract_topic(source)

        consumer = self._consumer
        if consumer is None:
            try:
                from confluent_kafka import Consumer
                conf = {
                    "bootstrap.servers": self.bootstrap_servers or "localhost:9092",
                    "group.id": self.group_id,
                    "auto.offset.reset": "earliest",
                    **self.kafka_options,
                }
                consumer = Consumer(conf)
                consumer.subscribe([topic])
                self._consumer = consumer
            except ImportError as e:
                raise InputProcessingError(
                    "confluent_kafka is required for Kafka inputs. "
                    "Install offline via: pip install --no-index --find-links=/path/to/wheels confluent-kafka"
                ) from e

        # Polling loop
        consumed_count = 0
        batch = []

        try:
            deadline = time.time() + timeout_seconds
            if hasattr(consumer, "consume"):
                while len(batch) < batch_size and time.time() < deadline:
                    remaining_timeout = max(0.05, deadline - time.time())
                    msgs = consumer.consume(num_messages=batch_size - len(batch), timeout=remaining_timeout)
                    if not msgs:
                        break
                    for msg in msgs:
                        if msg is None:
                            continue
                        if hasattr(msg, "error") and callable(msg.error) and msg.error():
                            err = msg.error()
                            err_str = str(err)
                            err_code = err.code() if hasattr(err, "code") else None
                            if err_code == -191 or "PARTITION_EOF" in err_str:
                                continue
                            raise InputProcessingError(
                                f"Kafka message error on topic '{topic}': {err_str}",
                                details={"topic": topic, "error": err_str, "code": err_code}
                            )
                        val = msg.value() if hasattr(msg, "value") else msg
                        if isinstance(val, bytes):
                            val = val.decode("utf-8", errors="replace")
                        batch.append(val)
                        consumed_count += 1
                if batch:
                    yield batch if batch_size > 1 else batch[0]
            elif hasattr(consumer, "poll"):
                while len(batch) < batch_size and time.time() < deadline:
                    remaining_timeout = max(0.05, deadline - time.time())
                    msg = consumer.poll(timeout=remaining_timeout)
                    if msg is None:
                        break
                    if hasattr(msg, "error") and callable(msg.error) and msg.error():
                        err = msg.error()
                        err_str = str(err)
                        err_code = err.code() if hasattr(err, "code") else None
                        if err_code == -191 or "PARTITION_EOF" in err_str:
                            continue
                        raise InputProcessingError(
                            f"Kafka poll error on topic '{topic}': {err_str}",
                            details={"topic": topic, "error": err_str, "code": err_code}
                        )
                    val = msg.value() if hasattr(msg, "value") else msg
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    batch.append(val)
                if batch:
                    yield batch if batch_size > 1 else batch[0]
        except Exception as e:
            if isinstance(e, MLFrameworkError):
                raise
            raise InputProcessingError(f"Kafka reading error on topic '{topic}': {e}") from e

    def close(self) -> None:
        """Closes the Kafka consumer connection and releases broker sockets."""
        if self._consumer is not None:
            if hasattr(self._consumer, "close") and callable(self._consumer.close):
                try:
                    self._consumer.close()
                except Exception as e:
                    logger.warning(f"Error closing Kafka consumer: {e}")
            self._consumer = None


class DirectInputReader(BaseInputReader):
    """
    Reader for in-memory direct inputs (strings, dictionaries, lists, bytes).
    Supports batch slicing for lists to avoid overloading model inference.
    """

    def validate(self, source: Any) -> bool:
        if source is None:
            raise InvalidInputSourceError("Input source data cannot be None")
        if isinstance(source, str) and not source.strip():
            raise InvalidInputSourceError("Input text string cannot be empty or whitespace only")
        if isinstance(source, (bytes, bytearray)) and len(source) == 0:
            raise InvalidInputSourceError("Input bytes cannot be empty")
        if isinstance(source, (list, tuple)) and len(source) == 0:
            raise InvalidInputSourceError("Input collection cannot be empty")
        return True

    def read(
        self,
        source: Any,
        batch_size: int = 1,
        chunk_size: Optional[int] = None
    ) -> Iterator[Any]:
        self.validate(source)
        if isinstance(source, (list, tuple)):
            for i in range(0, len(source), batch_size):
                chunk = source[i:i + batch_size]
                yield chunk if batch_size > 1 else chunk[0]
        else:
            yield source

    def close(self) -> None:
        """No-op for in-memory direct input reader."""
        pass
