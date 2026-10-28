"""
Input Processor implementation coordinating input validation, format detection,
and chunked reading across local files, object storage (SeaweedFS/MinIO), and Kafka.
"""

import os
import logging
from typing import Any, Dict, Optional, Iterator, Union

from ..interfaces.base import BaseInputProcessor, BaseInputReader
from ..config.model_config import InputConfig, StorageConfig
from ..results.execution_result import ExecutionContext
from ..errors.exceptions import (
    InputProcessingError,
    InvalidInputSourceError,
)
from .readers import (
    LocalFileReader,
    ObjectStorageReader,
    KafkaTopicReader,
    DirectInputReader,
)

logger = logging.getLogger("ml_framework.input.processor")


class InputProcessor(BaseInputProcessor):
    """
    Central input processor for the ML Framework.

    Orchestrates source validation, modality preprocessing, and chunked ingestion
    before expensive model loading or inference occurs.
    """

    def __init__(
        self,
        config: Optional[InputConfig] = None,
        storage_config: Optional[StorageConfig] = None,
        local_reader: Optional[BaseInputReader] = None,
        storage_reader: Optional[BaseInputReader] = None,
        kafka_reader: Optional[BaseInputReader] = None,
        direct_reader: Optional[BaseInputReader] = None,
    ):
        self.config = config or InputConfig()
        self.storage_config = storage_config or self.config.storage_config or StorageConfig()

        self.local_reader = local_reader or LocalFileReader(
            default_chunk_size=self.config.chunk_size or 65536,
            encoding=self.config.encoding
        )
        self.storage_reader = storage_reader or ObjectStorageReader(
            storage_config=self.storage_config,
            default_chunk_size=self.config.chunk_size or 65536
        )
        self.kafka_reader = kafka_reader or KafkaTopicReader(
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            group_id=self.config.kafka_group_id,
            kafka_options=self.config.kafka_options
        )
        self.direct_reader = direct_reader or DirectInputReader()

    def detect_source_type(self, source: Any) -> str:
        """
        Infers the source type from the source input structure or URI scheme.
        Returns: 'object_storage' | 'kafka' | 'file' | 'direct'
        """
        if source is None:
            raise InvalidInputSourceError("Input source cannot be None")

        # If caller explicitly configured a non-auto source_type, respect it
        if self.config and self.config.source_type and self.config.source_type != "auto":
            if self.config.source_type in {"file", "object_storage", "kafka", "direct"}:
                return self.config.source_type

        if isinstance(source, (str, bytes, os.PathLike)):
            if isinstance(source, (bytes, bytearray)):
                try:
                    source_str = os.fsdecode(source)
                except Exception:
                    return "direct"
            else:
                source_str = str(source)

            lower = source_str.lower()
            if lower.startswith(("s3://", "minio://", "seaweedfs://")):
                return "object_storage"
            if lower.startswith("kafka://"):
                return "kafka"
            if lower.startswith("file://"):
                return "file"
            if lower.startswith(("http://", "https://", "ftp://", "gcs://")):
                raise InvalidInputSourceError(
                    f"Unsupported URI scheme in input source: '{source_str}'. "
                    f"Supported schemes: file://, s3://, minio://, seaweedfs://, kafka://"
                )

            # Check if it is a local filesystem path
            if os.path.exists(source_str):
                return "file"

            KNOWN_EXTENSIONS = (
                ".txt", ".json", ".jsonl", ".csv", ".tsv",
                ".wav", ".mp3", ".flac", ".ogg", ".bin", ".onnx",
                ".parquet", ".xml", ".html"
            )
            if lower.endswith(KNOWN_EXTENSIONS):
                return "file"

            if source_str.startswith(("/", "./", "../", "~/", "\\")):
                return "file"
            if len(source_str) >= 3 and source_str[1:3] in {":\\", ":/"}:
                return "file"

            # Otherwise, treat as direct string payload
            return "direct"

        if isinstance(source, dict):
            if "kafka_topic" in source or "topic" in source:
                return "kafka"
            if "endpoint_url" in source or "bucket" in source:
                return "object_storage"
            return "direct"

        if isinstance(source, (list, tuple, bytes, bytearray)):
            return "direct"

        return "direct"

    def get_reader(self, source: Any) -> BaseInputReader:
        """Resolves the appropriate BaseInputReader for the given source."""
        source_type = self.detect_source_type(source)
        if source_type == "object_storage":
            return self.storage_reader
        elif source_type == "kafka":
            return self.kafka_reader
        elif source_type == "file":
            return self.local_reader
        else:
            return self.direct_reader

    def validate_source(self, source: Any) -> bool:
        """
        Validates source format and accessibility before expensive model loading.
        Raises InvalidInputSourceError or InputProcessingError if invalid.
        """
        if source is None:
            raise InvalidInputSourceError("Input source cannot be None")

        reader = self.get_reader(source)
        return reader.validate(source)

    def stream_input(
        self,
        source: Any,
        batch_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        as_bytes: bool = False
    ) -> Iterator[Any]:
        """
        Yields chunked or batched records from the source.
        Ensures memory efficiency for large files or object storage streams.
        """
        reader = self.get_reader(source)
        effective_batch = batch_size or self.config.batch_size
        effective_chunk = chunk_size or self.config.chunk_size

        if isinstance(reader, (LocalFileReader, ObjectStorageReader)):
            return reader.read(source, batch_size=effective_batch, chunk_size=effective_chunk, as_bytes=as_bytes)
        elif isinstance(reader, KafkaTopicReader):
            return reader.read(source, batch_size=effective_batch)
        else:
            return reader.read(source, batch_size=effective_batch, chunk_size=effective_chunk)

    def process(self, raw_input: Any, context: ExecutionContext) -> Any:
        """
        Prepares raw input for inference consumption.

        1. Validates input source or data.
        2. If input is a path or storage URI, resolves and reads the content.
        3. Sanitizes and structures input data for the model runner.
        """
        if raw_input is None:
            raise InvalidInputSourceError("Raw input data cannot be None")

        source_type = self.detect_source_type(raw_input)
        context.metadata["input_source_type"] = source_type

        # Validate source accessibility
        self.validate_source(raw_input)

        BINARY_EXTENSIONS = {
            ".wav", ".mp3", ".flac", ".ogg", ".bin", ".bytes", ".pcm", ".pt", ".onnx", ".raw"
        }

        if source_type == "file":
            reader = self.local_reader
            is_bin = False
            if isinstance(raw_input, (str, bytes, os.PathLike)):
                clean_path = self.local_reader._clean_path(raw_input)
                ext = os.path.splitext(clean_path.lower())[1]
                is_bin = ext in BINARY_EXTENSIONS

            # If unchunked binary file, read complete content
            if is_bin and self.config.chunk_size is None:
                path = self.local_reader._clean_path(raw_input)
                try:
                    with open(path, "rb") as f:
                        content = f.read()
                    if not content:
                        raise InputProcessingError(f"Input file '{raw_input}' was empty")
                    return content
                except OSError as e:
                    if isinstance(e, MLFrameworkError):
                        raise
                    raise InputProcessingError(f"Failed to read local file '{path}': {e}") from e

            for item in reader.read(
                raw_input,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size,
                as_bytes=is_bin
            ):
                return item
            raise InputProcessingError(f"Input file '{raw_input}' was empty")

        elif source_type == "object_storage":
            reader = self.storage_reader
            is_bin = False
            if isinstance(raw_input, str):
                ext = os.path.splitext(raw_input.lower().split("?")[0])[1]
                is_bin = ext in BINARY_EXTENSIONS

            # If unchunked binary storage object, read complete content
            if is_bin and self.config.chunk_size is None:
                _, bucket, key = self.storage_reader._parse_uri(str(raw_input))
                client = self.storage_reader._get_client()
                try:
                    resp = client.get_object(Bucket=bucket, Key=key)
                    body = resp.get("Body")
                    try:
                        content = body.read() if body else b""
                    finally:
                        if body and hasattr(body, "close"):
                            body.close()
                    if not content:
                        raise InputProcessingError(f"Object storage source '{raw_input}' was empty")
                    return content
                except Exception as e:
                    if isinstance(e, MLFrameworkError):
                        raise
                    raise StorageError(f"Failed to read object '{raw_input}': {e}") from e

            for item in reader.read(
                raw_input,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size,
                as_bytes=is_bin
            ):
                return item
            raise InputProcessingError(f"Object storage source '{raw_input}' was empty")

        elif source_type == "kafka":
            reader = self.kafka_reader
            timeout = float(self.config.custom_options.get("timeout_seconds", 5.0))
            for item in reader.read(raw_input, batch_size=self.config.batch_size, timeout_seconds=timeout):
                return item
            raise InputProcessingError(f"No messages received from Kafka source '{raw_input}'")

        else:
            # Direct in-memory input
            if isinstance(raw_input, str):
                cleaned = raw_input.strip()
                if not cleaned:
                    raise InvalidInputSourceError("Input text cannot be empty or whitespace only")
                return cleaned
            if isinstance(raw_input, (bytes, bytearray)) and len(raw_input) == 0:
                raise InvalidInputSourceError("Input bytes cannot be empty")
            return raw_input

    def close(self) -> None:
        """Closes all underlying readers and releases connection pools or sockets."""
        for reader in (self.local_reader, self.storage_reader, self.kafka_reader, self.direct_reader):
            if reader is not None and hasattr(reader, "close") and callable(reader.close):
                try:
                    reader.close()
                except Exception as e:
                    logger.warning(f"Error closing reader {reader}: {e}")

