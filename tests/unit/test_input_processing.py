"""
Unit tests for Input Processing: local files, object storage (SeaweedFS/MinIO),
Kafka, chunking, and validation before model loading.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import InputConfig, StorageConfig, FrameworkConfig
from ml_framework.input import (
    InputProcessor,
    LocalFileReader,
    ObjectStorageReader,
    KafkaTopicReader,
    DirectInputReader,
)
from ml_framework.results import ExecutionContext
from ml_framework.errors import (
    InputProcessingError,
    InvalidInputSourceError,
    StorageError,
    TransientStorageError,
    ConfigurationError,
)
from ml_framework.execution_manager import ExecutionManager


class TestLocalFileReader(unittest.TestCase):
    """Test suite for LocalFileReader with chunking and validation."""

    def setUp(self):
        self.reader = LocalFileReader(default_chunk_size=16)

    def test_valid_file_reading(self):
        """Test reading lines from a valid local file."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_path = f.name

        try:
            items = list(self.reader.read(temp_path, batch_size=2))
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0], ["line 1", "line 2"])
            self.assertEqual(items[1], ["line 3"])
        finally:
            os.remove(temp_path)

    def test_chunked_byte_reading(self):
        """Test streaming raw bytes in chunks to prevent buffering entire file in memory."""
        content = b"ABCDEFGHIJKLMNO_1234567890"  # 26 bytes
        with tempfile.NamedTemporaryFile("wb+", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            chunks = list(self.reader.read(temp_path, chunk_size=10, as_bytes=True))
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0], b"ABCDEFGHIJ")
            self.assertEqual(chunks[1], b"KLMNO_1234")
            self.assertEqual(chunks[2], b"567890")
            self.assertEqual(b"".join(chunks), content)
        finally:
            os.remove(temp_path)

    def test_nonexistent_file_rejection(self):
        """Test that missing file raises InvalidInputSourceError."""
        with self.assertRaises(InvalidInputSourceError) as cm:
            self.reader.validate("/non/existent/path/data.txt")
        self.assertIn("does not exist", str(cm.exception))

    def test_directory_source_rejection(self):
        """Test that directory path is rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(InvalidInputSourceError) as cm:
                self.reader.validate(tmp_dir)
            self.assertIn("expected a regular file", str(cm.exception))

    def test_file_uri_scheme_handling(self):
        """Test handling file:// URI scheme."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            f.write("content from URI")
            temp_path = f.name

        try:
            uri = f"file://{temp_path}"
            self.assertTrue(self.reader.validate(uri))
            items = list(self.reader.read(uri))
            self.assertEqual(items, ["content from URI"])
        finally:
            os.remove(temp_path)


class TestObjectStorageReader(unittest.TestCase):
    """Test suite for ObjectStorageReader supporting SeaweedFS, MinIO, and S3."""

    def test_uri_parsing(self):
        """Test URI parsing for s3://, minio://, and seaweedfs://."""
        reader = ObjectStorageReader()

        scheme, bucket, key = reader._parse_uri("seaweedfs://ml-data/inputs/audio.wav")
        self.assertEqual(scheme, "seaweedfs")
        self.assertEqual(bucket, "ml-data")
        self.assertEqual(key, "inputs/audio.wav")

        scheme, bucket, key = reader._parse_uri("minio://models-bucket/v1/config.json")
        self.assertEqual(scheme, "minio")
        self.assertEqual(bucket, "models-bucket")
        self.assertEqual(key, "v1/config.json")

    def test_unsupported_scheme_rejection(self):
        """Test unsupported schemes are rejected."""
        reader = ObjectStorageReader()
        with self.assertRaises(InvalidInputSourceError) as cm:
            reader._parse_uri("ftp://server/data.bin")
        self.assertIn("Unsupported object storage scheme", str(cm.exception))

    def test_missing_bucket_or_key_rejection(self):
        """Test invalid URI structure without bucket or key."""
        reader = ObjectStorageReader()
        with self.assertRaises(InvalidInputSourceError):
            reader._parse_uri("s3://")
        with self.assertRaises(InvalidInputSourceError):
            reader._parse_uri("s3://bucket-only")

    def test_mock_storage_streaming_read(self):
        """Test chunked streaming reading with mock S3/SeaweedFS/MinIO client."""
        mock_body = MagicMock()
        mock_body.iter_chunks.return_value = [b"chunk1_", b"chunk2_", b"chunk3"]

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_s3.head_object.return_value = {"ContentLength": 21}

        reader = ObjectStorageReader(s3_client=mock_s3)
        self.assertTrue(reader.validate("seaweedfs://audio-bucket/recordings/sample.wav"))

        chunks = list(reader.read(
            "seaweedfs://audio-bucket/recordings/sample.wav",
            chunk_size=7,
            as_bytes=True
        ))
        self.assertEqual(chunks, [b"chunk1_", b"chunk2_", b"chunk3"])
        mock_s3.get_object.assert_called_once_with(Bucket="audio-bucket", Key="recordings/sample.wav")

    def test_mock_storage_not_found(self):
        """Test simulated 404 / NoSuchKey raises InvalidInputSourceError."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = Exception("An error occurred (404) when calling HeadObject: Not Found")

        reader = ObjectStorageReader(s3_client=mock_s3)
        with self.assertRaises(InvalidInputSourceError) as cm:
            reader.validate("minio://bucket/missing-key.txt")
        self.assertIn("Object not found in storage", str(cm.exception))

    def test_mock_storage_transient_connection_error(self):
        """Test transient connection error raises TransientStorageError."""
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = Exception("Could not connect to the endpoint URL: http://minio:9000")

        reader = ObjectStorageReader(s3_client=mock_s3)
        with self.assertRaises(TransientStorageError) as cm:
            reader.validate("minio://bucket/file.txt")
        self.assertIn("Transient network or endpoint connection error", str(cm.exception))

    def test_mock_storage_client_verify_ssl_passed(self):
        """Test that ObjectStorageReader._get_client passes verify and use_ssl to boto3.client."""
        from unittest.mock import patch
        mock_boto3 = MagicMock()
        mock_botocore = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore": mock_botocore, "botocore.config": mock_botocore}):
            cfg = StorageConfig(
                endpoint_url="https://minio:9000",
                access_key="key",
                secret_key="secret",
                secure=True,
                verify_ssl=False
            )
            reader = ObjectStorageReader(storage_config=cfg)
            reader._get_client()
            mock_boto3.client.assert_called_once()
            _, kwargs = mock_boto3.client.call_args
            self.assertFalse(kwargs.get("verify"))
            self.assertTrue(kwargs.get("use_ssl"))
            self.assertEqual(kwargs.get("endpoint_url"), "https://minio:9000")


class TestKafkaTopicReader(unittest.TestCase):
    """Test suite for KafkaTopicReader using mock consumers."""

    def test_valid_topic_validation(self):
        """Test valid topic strings."""
        reader = KafkaTopicReader()
        self.assertTrue(reader.validate("ml.raw.events"))
        self.assertTrue(reader.validate("kafka://ml-inference-topic"))
        self.assertTrue(reader.validate({"kafka_topic": "topic_123"}))

    def test_invalid_topic_rejection(self):
        """Test empty or illegal topic names."""
        reader = KafkaTopicReader()
        with self.assertRaises(InvalidInputSourceError):
            reader.validate("")
        with self.assertRaises(InvalidInputSourceError):
            reader.validate("topic with spaces!")

    def test_mock_kafka_batch_consumption(self):
        """Test consuming batch of messages from mock Kafka consumer."""
        mock_consumer = MagicMock()

        class MockMsg:
            def __init__(self, val):
                self._val = val
            def value(self):
                return self._val

        mock_consumer.consume.return_value = [
            MockMsg(b"message 1"),
            MockMsg(b"message 2"),
        ]

        reader = KafkaTopicReader(consumer=mock_consumer)
        batches = list(reader.read("kafka://inference-input", batch_size=2))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], ["message 1", "message 2"])


class TestInputProcessorOrchestration(unittest.TestCase):
    """Test suite for high-level InputProcessor."""

    def test_source_detection(self):
        """Test automatic source type detection."""
        proc = InputProcessor()
        self.assertEqual(proc.detect_source_type("seaweedfs://b/key"), "object_storage")
        self.assertEqual(proc.detect_source_type("minio://b/key"), "object_storage")
        self.assertEqual(proc.detect_source_type("s3://b/key"), "object_storage")
        self.assertEqual(proc.detect_source_type("kafka://topic-a"), "kafka")
        self.assertEqual(proc.detect_source_type("/path/to/data.txt"), "file")
        self.assertEqual(proc.detect_source_type("Plain text input"), "direct")
        self.assertEqual(proc.detect_source_type({"key": "value"}), "direct")
        self.assertEqual(proc.detect_source_type({"topic": "stream-a"}), "kafka")

    def test_unsupported_scheme_rejection(self):
        """Test unsupported schemes (ftp://, gcs://, http://) raise InvalidInputSourceError."""
        proc = InputProcessor()
        with self.assertRaises(InvalidInputSourceError) as cm:
            proc.detect_source_type("ftp://example.com/data.txt")
        self.assertIn("Unsupported URI scheme", str(cm.exception))

    def test_empty_string_rejection(self):
        """Test empty string input raises InvalidInputSourceError."""
        proc = InputProcessor()
        ctx = ExecutionContext()
        with self.assertRaises(InvalidInputSourceError):
            proc.process("   ", ctx)

    def test_pre_load_validation_and_graceful_shutdown(self):
        """
        Verify that invalid input configuration is rejected BEFORE starting expensive operations,
        and manager cleans up and transitions to STOPPED/FAILED gracefully.
        """
        cfg = FrameworkConfig(
            input_source="/non/existent/critical/dataset.jsonl",
            model_name="expensive_llm",
        )
        manager = ExecutionManager(config=cfg)

        with self.assertRaises(InvalidInputSourceError):
            manager.initialize()

        # Manager should have gracefully cleaned up and set state to STOPPED/FAILED
        self.assertIn(manager.state.value, ["STOPPED", "FAILED"])
        self.assertFalse(manager.is_ready)

    def test_audio_binary_file_reading_in_processor(self):
        """Test that binary and audio files are read as raw bytes without string decoding."""
        proc = InputProcessor()
        ctx = ExecutionContext()
        raw_audio_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"

        with tempfile.NamedTemporaryFile("wb+", suffix=".wav", delete=False) as f:
            f.write(raw_audio_bytes)
            temp_wav = f.name

        try:
            result = proc.process(temp_wav, ctx)
            self.assertIsInstance(result, bytes)
            self.assertEqual(result, raw_audio_bytes)
        finally:
            os.remove(temp_wav)

    def test_object_storage_streaming_body_closed_on_read(self):
        """Test that botocore StreamingBody is properly closed after read finishes."""
        mock_body = MagicMock()
        mock_body.iter_lines.return_value = [b"line1\n", b"line2\n"]

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}

        reader = ObjectStorageReader(s3_client=mock_s3)
        items = list(reader.read("s3://test-bucket/data.txt", batch_size=2))

        self.assertEqual(items, [["line1", "line2"]])
        mock_body.close.assert_called_once()

    def test_object_storage_close_closes_client(self):
        """Test that ObjectStorageReader.close() closes the S3 client."""
        mock_s3 = MagicMock()
        reader = ObjectStorageReader(s3_client=mock_s3)

        reader.close()
        mock_s3.close.assert_called_once()
        self.assertIsNone(reader._client)

    def test_kafka_reader_close_closes_consumer(self):
        """Test that KafkaTopicReader.close() closes the consumer and releases resources."""
        mock_consumer = MagicMock()
        reader = KafkaTopicReader(consumer=mock_consumer)

        reader.close()
        mock_consumer.close.assert_called_once()
        self.assertIsNone(reader._consumer)

    def test_kafka_message_error_handling(self):
        """Test that Kafka consumer message errors raise InputProcessingError, while partition EOF is ignored."""
        mock_consumer = MagicMock()

        class MockErr:
            def __init__(self, code, msg):
                self._code = code
                self._msg = msg
            def code(self):
                return self._code
            def __str__(self):
                return self._msg

        class MockErrorMsg:
            def __init__(self, err):
                self._err = err
            def error(self):
                return self._err
            def value(self):
                return None

        # 1. Non-fatal partition EOF should be skipped
        eof_msg = MockErrorMsg(MockErr(-191, "PARTITION_EOF"))
        mock_consumer.consume.return_value = [eof_msg]

        reader = KafkaTopicReader(consumer=mock_consumer)
        items = list(reader.read("kafka://test-topic", batch_size=1))
        self.assertEqual(items, [])

        # 2. Fatal broker error should raise InputProcessingError
        fatal_msg = MockErrorMsg(MockErr(1, "BrokerNotAvailable"))
        mock_consumer.consume.return_value = [fatal_msg]

        with self.assertRaises(InputProcessingError) as cm:
            list(reader.read("kafka://test-topic", batch_size=1))
        self.assertIn("Kafka message error on topic", str(cm.exception))

    def test_input_processor_close_propagates_to_readers(self):
        """Test that InputProcessor.close() calls close() on all underlying readers."""
        mock_local = MagicMock()
        mock_storage = MagicMock()
        mock_kafka = MagicMock()
        mock_direct = MagicMock()

        proc = InputProcessor(
            local_reader=mock_local,
            storage_reader=mock_storage,
            kafka_reader=mock_kafka,
            direct_reader=mock_direct,
        )

        proc.close()
        mock_local.close.assert_called_once()
        mock_storage.close.assert_called_once()
        mock_kafka.close.assert_called_once()
        mock_direct.close.assert_called_once()

    def test_execution_manager_shutdown_closes_input_processor(self):
        """Test that ExecutionManager.shutdown() closes the injected input processor."""
        mock_proc = MagicMock()
        manager = ExecutionManager(input_processor=mock_proc)
        manager.initialize()
        manager.shutdown()

        mock_proc.close.assert_called_once()

    def test_slash_in_natural_language_not_treated_as_file(self):
        """Test natural language containing slashes is correctly resolved to direct input."""
        proc = InputProcessor()
        ctx = ExecutionContext()

        self.assertEqual(proc.detect_source_type("She said yes/no to the proposal."), "direct")
        self.assertEqual(proc.detect_source_type("2025/2026 report and/or summary"), "direct")
        self.assertEqual(proc.detect_source_type("Apple / Microsoft partnership"), "direct")

        result = proc.process("She said yes/no to the proposal.", ctx)
        self.assertEqual(result, "She said yes/no to the proposal.")

    def test_explicit_source_type_override(self):
        """Test that an explicitly configured source_type overrides automatic inference."""
        cfg = InputConfig(source_type="direct")
        proc = InputProcessor(config=cfg)
        self.assertEqual(proc.detect_source_type("/path/treated/as/direct"), "direct")

    def test_large_binary_file_reading_without_truncation(self):
        """Test that binary files larger than default chunk size (64KB) are not truncated in process()."""
        proc = InputProcessor()
        ctx = ExecutionContext()
        large_bytes = b"WAVE" + (b"\x00" * 100000)  # 100,004 bytes

        with tempfile.NamedTemporaryFile("wb+", suffix=".wav", delete=False) as f:
            f.write(large_bytes)
            temp_path = f.name

        try:
            result = proc.process(temp_path, ctx)
            self.assertEqual(len(result), len(large_bytes))
            self.assertEqual(result, large_bytes)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_empty_file_raises_input_processing_error(self):
        """Test that zero-byte files raise InputProcessingError in process()."""
        proc = InputProcessor()
        ctx = ExecutionContext()

        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
            temp_path = f.name

        try:
            with self.assertRaises(InputProcessingError) as cm:
                proc.process(temp_path, ctx)
            self.assertIn("was empty", str(cm.exception))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_mock_object_storage_large_binary_read(self):
        """Test unchunked binary reading from object storage preserves full payload."""
        large_bytes = b"RIFF" + (b"\x01" * 80000)
        mock_body = MagicMock()
        mock_body.read.return_value = large_bytes

        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": mock_body}
        mock_s3.head_object.return_value = {"ContentLength": len(large_bytes)}

        storage_reader = ObjectStorageReader(s3_client=mock_s3)
        proc = InputProcessor(storage_reader=storage_reader)
        ctx = ExecutionContext()

        result = proc.process("seaweedfs://audio/test.wav", ctx)
        self.assertEqual(len(result), len(large_bytes))
        self.assertEqual(result, large_bytes)

    def test_bytes_path_decoding_in_local_file_reader(self):
        """Test passing bytes path to LocalFileReader._clean_path()."""
        reader = LocalFileReader()
        cleaned = reader._clean_path(b"/tmp/test_path.txt")
        self.assertEqual(cleaned, "/tmp/test_path.txt")

    def test_kafka_poll_batching(self):
        """Test KafkaTopicReader polling multiple messages in a batch when using poll()."""
        mock_consumer = MagicMock(spec=["poll", "close"])

        class MockMsg:
            def __init__(self, val):
                self._val = val
            def error(self):
                return None
            def value(self):
                return self._val

        mock_consumer.poll.side_effect = [
            MockMsg(b"batch 1"),
            MockMsg(b"batch 2"),
            MockMsg(b"batch 3"),
            None,
        ]

        reader = KafkaTopicReader(consumer=mock_consumer)
        batches = list(reader.read("kafka://poll-topic", batch_size=3))

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], ["batch 1", "batch 2", "batch 3"])


class TestDirectInputReader(unittest.TestCase):
    """Test suite for DirectInputReader."""

    def setUp(self):
        self.reader = DirectInputReader()

    def test_valid_direct_inputs(self):
        """Test validating and reading valid strings, bytes, dicts, and lists."""
        self.assertTrue(self.reader.validate("valid text"))
        self.assertTrue(self.reader.validate(b"valid bytes"))
        self.assertTrue(self.reader.validate({"key": "val"}))
        self.assertTrue(self.reader.validate(["item 1", "item 2"]))

        items = list(self.reader.read("single text"))
        self.assertEqual(items, ["single text"])

    def test_empty_bytes_rejection(self):
        """Test that empty bytes b'' raise InvalidInputSourceError."""
        with self.assertRaises(InvalidInputSourceError) as cm:
            self.reader.validate(b"")
        self.assertIn("Input bytes cannot be empty", str(cm.exception))

    def test_empty_string_rejection(self):
        """Test that empty string raises InvalidInputSourceError."""
        with self.assertRaises(InvalidInputSourceError) as cm:
            self.reader.validate("   ")
        self.assertIn("cannot be empty or whitespace only", str(cm.exception))

    def test_empty_collection_rejection(self):
        """Test that empty list raises InvalidInputSourceError."""
        with self.assertRaises(InvalidInputSourceError) as cm:
            self.reader.validate([])
        self.assertIn("Input collection cannot be empty", str(cm.exception))

    def test_none_rejection(self):
        """Test that None raises InvalidInputSourceError."""
        with self.assertRaises(InvalidInputSourceError) as cm:
            self.reader.validate(None)
        self.assertIn("cannot be None", str(cm.exception))

    def test_batch_slicing_on_collection(self):
        """Test slicing in-memory collections by batch_size."""
        data = ["a", "b", "c", "d", "e"]
        batches = list(self.reader.read(data, batch_size=2))
        self.assertEqual(batches, [["a", "b"], ["c", "d"], ["e"]])


class TestInputConfigValidation(unittest.TestCase):
    """Test suite for InputConfig schema validation."""

    def test_valid_source_types(self):
        """Test all valid source types pass validation."""
        for st in ("auto", "file", "object_storage", "kafka", "direct"):
            cfg = InputConfig(source_type=st)
            cfg.validate()

    def test_invalid_source_type_rejection(self):
        """Test unsupported source_type raises ConfigurationError."""
        cfg = InputConfig(source_type="unsupported_network_socket")
        with self.assertRaises(ConfigurationError) as cm:
            cfg.validate()
        self.assertIn("Invalid source_type 'unsupported_network_socket'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

