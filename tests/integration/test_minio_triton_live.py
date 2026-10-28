"""
Opt-in Live Integration Tests for MinIO-to-Triton Model Retrieval and Serving.

Verifies the complete live flow from MinIO S3 object storage to NVIDIA Triton Inference Server:
1. MinIO model artifact selection via configurable bucket and object key.
2. Artifact retrieval from live MinIO service using authorized credentials.
3. Intact .nemo archive staging to Triton model staging directory.
4. Triton server readiness probe and model readiness verification.
5. End-to-end audio transcription inference on test3.wav through Triton API.
6. Verification that production inference is executed exclusively by Triton.

Execution (when company network connectivity to MinIO is available):
    TEST_LIVE_MINIO=1 \
    MINIO_ENDPOINT="http://minio.team.internal:9000" \
    MINIO_ACCESS_KEY="<ACCESS_KEY>" \
    MINIO_SECRET_KEY="<SECRET_KEY>" \
    MINIO_MODEL_BUCKET="speech-data" \
    MINIO_MODEL_KEY="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo" \
    TRITON_SERVER_URL="http://localhost:8010" \
    python3 -m unittest -v tests/integration/test_minio_triton_live.py

Safety & Isolation Guarantees:
- Skipped by default unless TEST_LIVE_MINIO=1 is explicitly set.
- Never prints or leaks credentials or secrets.
- Never deletes shared buckets or live model objects.
- Distinguishes unit test mock results from actual live network execution.
"""

import os
import sys
import unittest
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import ml, FrameworkConfig, StorageConfig, TritonConfig, ExecutionState
from ml_framework.inference.triton_client import TritonClient
from ml_framework.inference.triton_engine import TritonInferenceEngine
from ml_framework.models.model_loader import ModelLoader
from ml_framework.config.model_config import parse_env_file

TRITON_TEST_URL = os.environ.get("TRITON_SERVER_URL", "http://localhost:8010")
TEST_AUDIO_PATH = "/tmp/poc_nemo_audio/test3.wav"


def _load_env_vars() -> None:
    """Loads variables from local .env into os.environ if not already present."""
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.is_file():
        parsed = parse_env_file(env_file)
        for k, v in parsed.items():
            if k not in os.environ:
                os.environ[k] = v


_load_env_vars()


def is_live_minio_enabled() -> bool:
    """Checks whether live MinIO integration testing is explicitly enabled."""
    return os.environ.get("TEST_LIVE_MINIO", "").strip().lower() in ("1", "true", "yes")


@unittest.skipUnless(
    is_live_minio_enabled(),
    "Live MinIO-to-Triton testing disabled. Set TEST_LIVE_MINIO=1 when company network is available."
)
class TestMinIOTritonLiveIntegration(unittest.TestCase):
    """
    Live integration test suite communicating with active MinIO storage and Triton server.
    """

    storage_cfg = None
    triton_client = None
    triton_ready = False
    bucket = None
    key = None

    @classmethod
    def setUpClass(cls):
        # 1. Resolve storage config
        cls.storage_cfg = StorageConfig.from_env()
        cls.bucket = (
            cls.storage_cfg.model_bucket
            or os.environ.get("MINIO_MODEL_BUCKET")
            or os.environ.get("S3_MODEL_BUCKET")
            or os.environ.get("MINIO_TEST_BUCKET")
        )
        cls.key = (
            cls.storage_cfg.model_key
            or os.environ.get("MINIO_MODEL_KEY")
            or os.environ.get("S3_MODEL_KEY")
            or "model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo"
        )

        # 2. Check storage credentials
        cls.storage_cfg.validate_live_config(bucket=cls.bucket)

        # 3. Check Triton server readiness
        cls.triton_client = TritonClient(endpoint_url=TRITON_TEST_URL, timeout_seconds=10.0)
        cls.triton_ready = cls.triton_client.is_server_ready()

    def setUp(self):
        if not self.triton_ready:
            self.skipTest(f"Triton server at {TRITON_TEST_URL} is not reachable. Skipping live MinIO-to-Triton test.")
        if not self.bucket or not self.key:
            self.skipTest("MINIO_MODEL_BUCKET and MINIO_MODEL_KEY must be configured for live test.")

    def test_live_minio_retrieval_and_triton_readiness(self):
        """
        Verify live retrieval of .nemo model artifact from MinIO, staging into Triton,
        and verification that Triton reaches READY with the retrieved model.
        """
        loader = ModelLoader(storage_config=self.storage_cfg)
        try:
            staging_dir = os.environ.get("TRITON_STAGING_DIR", "/tmp/poc_nemo_model")
            triton_cfg = TritonConfig(
                endpoint_url=TRITON_TEST_URL,
                model_name="nemo_conformer_python",
                backend="python",
                staging_dir=staging_dir,
                check_model_ready=True,
            )

            model_uri = f"s3://{self.bucket}/{self.key.lstrip('/')}"
            metadata = loader.load_for_triton(
                model_source=model_uri,
                config=FrameworkConfig(
                    model_bucket=self.bucket,
                    model_key=self.key,
                    storage_config=self.storage_cfg,
                    triton_config=triton_cfg,
                ),
            )

            self.assertEqual(metadata["model_name"], "nemo_conformer_python")
            self.assertEqual(metadata["framework"], "nemo")
            self.assertEqual(metadata["triton_backend"], "python")
            self.assertTrue(os.path.isfile(metadata["staged_path"]))

            # Verify Triton reports model is READY
            self.assertTrue(self.triton_client.is_model_ready("nemo_conformer_python"))
        finally:
            loader.close()

    def test_live_minio_to_triton_end_to_end_inference(self):
        """
        Verify complete user workflow:
        ml.load(model_bucket=..., model_key=...) -> retrieve from MinIO -> stage -> execute(test3.wav) via Triton
        """
        if not os.path.isfile(TEST_AUDIO_PATH):
            self.skipTest(f"Test audio file '{TEST_AUDIO_PATH}' not found.")

        staging_dir = os.environ.get("TRITON_STAGING_DIR", "/tmp/poc_nemo_model")
        cfg = FrameworkConfig(
            model_bucket=self.bucket,
            model_key=self.key,
            timeout_seconds=60.0,
            storage_config=self.storage_cfg,
            triton_config=TritonConfig(
                endpoint_url=TRITON_TEST_URL,
                model_name="nemo_conformer_python",
                backend="python",
                staging_dir=staging_dir,
                check_model_ready=True,
                timeout_seconds=60.0,
            ),
        )

        manager = ml.load(config=cfg)
        self.assertTrue(manager.is_ready)
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertIsInstance(manager.inference_engine, TritonInferenceEngine)

        try:
            result = manager.execute(TEST_AUDIO_PATH)
            self.assertTrue(result.success, f"Inference failed: {result.error}")
            self.assertIsNotNone(result.output)
            self.assertIn("transcription", result.output)

            transcript = result.output["transcription"].strip()
            self.assertIn("one two", transcript)
            self.assertIn("three", transcript)

            # Verify context metadata reflects Triton execution
            self.assertIn("triton_model_name", result.context.metadata)
            self.assertEqual(result.context.metadata["triton_model_name"], "nemo_conformer_python")
        finally:
            manager.unload()
            manager.stop()


if __name__ == "__main__":
    unittest.main()
