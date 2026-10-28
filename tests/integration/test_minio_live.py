"""
Opt-in Live Integration Tests for MinIO / S3-compatible Object Storage.

These tests execute against actual running MinIO / SeaweedFS / S3 infrastructure
when explicitly opted into via environment variables.

Execution:
    TEST_LIVE_MINIO=1 \
    MINIO_ENDPOINT="http://minio.team.internal:9000" \
    MINIO_ACCESS_KEY="<ACCESS_KEY>" \
    MINIO_SECRET_KEY="<SECRET_KEY>" \
    MINIO_TEST_BUCKET="ml-framework-integration-test" \
    python3 -m unittest -v tests/integration/test_minio_live.py

Safety & Isolation Guarantees:
- Tests ONLY run when TEST_LIVE_MINIO=1 is set.
- All test fixtures are uploaded under an isolated prefix: ml-framework-live-tests/<uuid>/
- Only test objects explicitly created during test execution are deleted in tearDown.
- Never deletes shared buckets or unrelated objects in the storage bucket.
- Fails clearly if enabled but boto3 or service connectivity is unavailable.
"""

import os
import sys
import uuid
import unittest
import importlib.util
from pathlib import Path
from typing import List

# Ensure src/ is on python path
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config.model_config import StorageConfig
from ml_framework.input.readers import ObjectStorageReader
from ml_framework.input.processor import InputProcessor
from ml_framework.results.execution_result import ExecutionContext
from ml_framework.errors.exceptions import (
    InvalidInputSourceError,
    ConfigurationError,
)

HAS_BOTO3 = importlib.util.find_spec("boto3") is not None


def _load_test_env() -> None:
    """
    Explicitly loads local .env variables into os.environ for integration test execution only.
    Does not overwrite environment variables already set in the shell.
    Uses standard library only (no third-party dependencies).
    """
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_file.is_file():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
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
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_test_env()


def is_minio_enabled() -> bool:
    """Checks whether live MinIO integration testing is explicitly enabled."""
    return os.environ.get("TEST_LIVE_MINIO", "").strip().lower() in ("1", "true", "yes")


@unittest.skipUnless(is_minio_enabled(), "Live MinIO testing disabled. Set TEST_LIVE_MINIO=1 to enable.")
class TestMinIOLiveIntegration(unittest.TestCase):
    """
    Live integration test suite verifying ObjectStorageReader and InputProcessor
    against a live MinIO / S3 service.
    """

    s3 = None
    storage_config = None
    bucket = None

    @classmethod
    def setUpClass(cls):
        # 1. Check client library dependency
        if not HAS_BOTO3:
            raise AssertionError(
                "TEST_LIVE_MINIO is enabled, but 'boto3' is not installed in the active Python environment. "
                "Install it via: pip install '.[storage]' or pip install boto3"
            )

        # 2. Check and validate required configuration
        cls.storage_config = StorageConfig.from_dict(dict(os.environ))
        cls.bucket = os.environ.get("MINIO_TEST_BUCKET") or os.environ.get("S3_TEST_BUCKET")
        try:
            cls.storage_config.validate_live_config(bucket=cls.bucket)
        except ConfigurationError as cfg_err:
            raise AssertionError(
                f"TEST_LIVE_MINIO is enabled, but configuration is invalid: {cfg_err}"
            ) from cfg_err

        import boto3
        from botocore.config import Config

        boto_cfg = Config(signature_version="s3v4", s3={"addressing_style": "path"})
        cls.s3 = boto3.client(
            "s3",
            endpoint_url=cls.storage_config.endpoint_url,
            aws_access_key_id=cls.storage_config.access_key,
            aws_secret_access_key=cls.storage_config.secret_key,
            region_name=cls.storage_config.region,
            use_ssl=cls.storage_config.secure,
            verify=cls.storage_config.verify_ssl,
            config=boto_cfg,
        )

        # 3. Verify connectivity and bucket availability
        try:
            cls.s3.head_bucket(Bucket=cls.bucket)
        except Exception as e:
            err_str = str(e).lower()
            if "404" in err_str or "notfound" in err_str or "nosuchbucket" in err_str:
                # Attempt dedicated bucket creation if authorized
                try:
                    cls.s3.create_bucket(Bucket=cls.bucket)
                except Exception as create_err:
                    raise AssertionError(
                        f"Live MinIO test bucket '{cls.bucket}' does not exist on {cls.storage_config.endpoint_url} "
                        f"and could not be created automatically: {create_err}. "
                        f"Please create the dedicated bucket before running live tests: mc mb <alias>/{cls.bucket}"
                    ) from create_err
            else:
                raise AssertionError(
                    f"Could not connect to live MinIO service at '{cls.storage_config.endpoint_url}': {e}. "
                    f"Verify that the service is running, endpoint URL is correct, and network access is permitted."
                ) from e

    def setUp(self):
        self.created_keys: List[str] = []
        self.test_run_id = uuid.uuid4().hex[:8]
        self.key_prefix = f"ml-framework-live-tests/{self.test_run_id}"

    def tearDown(self):
        # Strict isolation: Clean up only test objects created by this specific test
        for key in self.created_keys:
            try:
                self.s3.delete_object(Bucket=self.bucket, Key=key)
            except Exception:
                pass

    def test_live_connection_and_text_read(self):
        """Verifies successful connection, validation, and line reading from real MinIO."""
        key = f"{self.key_prefix}/sample_text.txt"
        fixture_lines = [
            "ML Framework MinIO Live Verification",
            "Testing line 2 of text ingestion",
            "Final line of determinism",
        ]
        fixture_payload = "\n".join(fixture_lines) + "\n"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=fixture_payload.encode("utf-8")
        )
        self.created_keys.append(key)

        uri = f"minio://{self.bucket}/{key}"
        reader = ObjectStorageReader(storage_config=self.storage_config)

        # 1. Validate object source accessibility
        is_valid = reader.validate(uri)
        self.assertTrue(is_valid, f"Expected reader.validate({uri}) to return True")

        # 2. Read lines with batch_size=1
        read_lines = list(reader.read(uri, batch_size=1))
        self.assertEqual(read_lines, fixture_lines)

        reader.close()

    def test_live_missing_object_error_handling(self):
        """Verifies that accessing a non-existent object raises InvalidInputSourceError."""
        non_existent_key = f"{self.key_prefix}/non_existent_file_{uuid.uuid4().hex}.txt"
        uri = f"minio://{self.bucket}/{non_existent_key}"

        reader = ObjectStorageReader(storage_config=self.storage_config)

        # Validating non-existent object must raise InvalidInputSourceError
        with self.assertRaises(InvalidInputSourceError) as ctx:
            reader.validate(uri)
        self.assertIn("not found", str(ctx.exception).lower())

        # Reading non-existent object must also raise InvalidInputSourceError
        with self.assertRaises(InvalidInputSourceError):
            list(reader.read(uri))

        reader.close()

    def test_live_binary_content_preservation(self):
        """Verifies byte-exact binary reading without truncation or corruption."""
        key = f"{self.key_prefix}/test_audio.bin"
        # Deterministic 16 KB binary sequence covering all 0..255 byte values
        fixture_bytes = bytes([i % 256 for i in range(16384)])

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=fixture_bytes
        )
        self.created_keys.append(key)

        uri = f"minio://{self.bucket}/{key}"

        # Test chunked byte reading via ObjectStorageReader
        reader = ObjectStorageReader(storage_config=self.storage_config)
        chunks = list(reader.read(uri, as_bytes=True, chunk_size=4096))
        assembled_bytes = b"".join(chunks)

        self.assertEqual(len(assembled_bytes), len(fixture_bytes))
        self.assertEqual(assembled_bytes, fixture_bytes)
        reader.close()

        # Test end-to-end reading via InputProcessor.process()
        processor = InputProcessor(storage_config=self.storage_config)
        processed_data = processor.process(uri, context=ExecutionContext())
        self.assertEqual(len(processed_data), len(fixture_bytes))
        self.assertEqual(processed_data, fixture_bytes)
        processor.close()

    def test_live_input_processor_streaming(self):
        """Verifies InputProcessor.stream_input yields all lines from MinIO."""
        key = f"{self.key_prefix}/streaming_data.jsonl"
        lines = [f'{{"record_id": {i}, "status": "active"}}' for i in range(5)]
        payload = "\n".join(lines) + "\n"

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=payload.encode("utf-8")
        )
        self.created_keys.append(key)

        uri = f"minio://{self.bucket}/{key}"
        processor = InputProcessor(storage_config=self.storage_config)

        streamed = list(processor.stream_input(uri, batch_size=1))
        self.assertEqual(streamed, lines)
        processor.close()

    def test_live_resource_cleanup(self):
        """Verifies that test cleanup deletes only the test fixture and keeps bucket intact."""
        key = f"{self.key_prefix}/cleanup_verification.txt"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=b"Temporary cleanup test object"
        )
        self.created_keys.append(key)

        reader = ObjectStorageReader(storage_config=self.storage_config)
        self.assertTrue(reader.validate(f"minio://{self.bucket}/{key}"))
        reader.close()

        # Explicitly delete the object
        self.s3.delete_object(Bucket=self.bucket, Key=key)
        self.created_keys.remove(key)

        # Confirm object is gone
        with self.assertRaises(InvalidInputSourceError):
            reader.validate(f"minio://{self.bucket}/{key}")

        # Confirm bucket itself is unaffected
        self.s3.head_bucket(Bucket=self.bucket)


if __name__ == "__main__":
    unittest.main()
