"""
Opt-in Live Integration Tests for Multi-Model MinIO Selection and Triton Serving.

Tests configuration-driven model selection, MinIO artifact retrieval (both single-file
checkpoints and multi-file directories), format detection, Triton serving, and dynamic
model switching without Python source modification.

Execution (when company network connectivity to MinIO is available):
    TEST_LIVE_MINIO=1 \
    MINIO_MODEL_BUCKET="speech-data" \
    TRITON_SERVER_URL="http://localhost:8010" \
    PYTHONPATH=src:/tmp/poc_nemo_pkgs/site-packages \
    python3 -m unittest -v tests/integration/test_multimodel_minio_live.py

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
from ml_framework.errors.exceptions import (
    ModelFormatError,
    DependencyMissingError,
    InputProcessingError,
)

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
    "Live MinIO multi-model testing disabled. Set TEST_LIVE_MINIO=1 when company network is available."
)
class TestMultiModelMinIOLiveIntegration(unittest.TestCase):
    """
    Live integration test suite communicating with active MinIO storage and Triton server
    across multiple model families.
    """

    storage_cfg = None
    triton_client = None
    triton_ready = False
    bucket = None

    @classmethod
    def setUpClass(cls):
        cls.storage_cfg = StorageConfig.from_env()
        cls.bucket = (
            cls.storage_cfg.model_bucket
            or os.environ.get("MINIO_MODEL_BUCKET")
            or os.environ.get("S3_MODEL_BUCKET")
            or "speech-data"
        )
        cls.storage_cfg.validate_live_config(bucket=cls.bucket)
        cls.triton_client = TritonClient(endpoint_url=TRITON_TEST_URL, timeout_seconds=10.0)
        cls.triton_ready = cls.triton_client.is_server_ready()

    def setUp(self):
        if not self.triton_ready:
            self.skipTest(f"Triton server at {TRITON_TEST_URL} is not reachable. Skipping live multi-model test.")
        if not self.bucket:
            self.skipTest("MINIO_MODEL_BUCKET must be configured for live test.")

    def test_live_nemo_conformer_transcription_e2e(self):
        """
        Family 2 (NeMo Conformer-CTC):
        Verify configuration-driven selection, live MinIO download, Triton staging,
        and real-audio transcription inference satisfying the output contract.
        """
        if not os.path.isfile(TEST_AUDIO_PATH):
            self.skipTest(f"Test audio file '{TEST_AUDIO_PATH}' not found.")

        staging_dir = os.environ.get("TRITON_STAGING_DIR", "/tmp/poc_nemo_model")
        cfg = FrameworkConfig(
            model_bucket=self.bucket,
            model_key="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo",
            task="transcription",
            input_type="audio",
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
            # Verify transcription contract
            self.assertEqual(result.output.get("task"), "transcription")
            self.assertIn("transcription", result.output)
            self.assertIn("text", result.output)
            transcript = result.output["transcription"].strip()
            self.assertTrue(len(transcript) > 0)
            self.assertIn("one two", transcript)
        finally:
            manager.unload()
            manager.stop()

    def test_live_multifile_directory_retrieval_and_structure_preservation(self):
        """
        Family 6 (DeepFilterNet3) / Multi-file Model Directory:
        Verify live MinIO download of a multi-file model directory preserves
        the relative directory structure (e.g. checkpoints subfolder and config).
        """
        loader = ModelLoader(storage_config=self.storage_cfg)
        try:
            staging_dir = "/tmp/poc_multimodel_staging"
            triton_cfg = TritonConfig(
                endpoint_url=TRITON_TEST_URL,
                model_name="deepfilternet3",
                backend="python",
                staging_dir=staging_dir,
                check_model_ready=False,
            )

            # DeepFilterNet3 directory in speech-data bucket
            model_dir = "model/Denoiser/DeepFilterNet3"
            model_uri = f"s3://{self.bucket}/{model_dir}"
            cfg = FrameworkConfig(
                model_bucket=self.bucket,
                model_dir=model_dir,
                task="denoising",
                input_type="audio",
                storage_config=self.storage_cfg,
                triton_config=triton_cfg,
            )

            # 1. Verify MinIO multi-file directory download preserves relative hierarchy
            downloaded_path = loader.retrieve_model_artifact(model_uri, cfg)
            self.assertTrue(os.path.isdir(downloaded_path))
            self.assertTrue(os.path.exists(os.path.join(downloaded_path, "config.ini")))
            checkpoints_dir = os.path.join(downloaded_path, "checkpoints")
            self.assertTrue(
                os.path.isdir(checkpoints_dir) or any("model" in f for f in os.listdir(downloaded_path))
            )

            # Verify format detection on preserved directory
            from ml_framework.models.format_detector import FormatDetector
            fw, arch, mtype = FormatDetector().detect(downloaded_path)
            self.assertEqual(fw, "deepfilternet")
            self.assertEqual(arch, "DeepFilterNet3")
            self.assertEqual(mtype, "deepfilternet3")

            # 2. Verify load_for_triton correctly guards with DependencyMissingError
            with self.assertRaises(DependencyMissingError) as ctx:
                loader.load_for_triton(model_source=model_uri, config=cfg)
            self.assertIn("deepfilternet", str(ctx.exception))
        finally:
            loader.close()

    def test_live_model_switching_and_backend_compatibility_guard(self):
        """
        Verify dynamic model switching from NeMo Conformer to another MinIO model
        (Wav2Vec2), confirming that the framework cleanly detects backend incompatibility
        with the active ASR backend rather than silently falling back to local inference.
        """
        staging_dir = os.environ.get("TRITON_STAGING_DIR", "/tmp/poc_nemo_model")
        cfg = FrameworkConfig(
            model_bucket=self.bucket,
            model_key="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo",
            task="transcription",
            input_type="audio",
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

        try:
            # Attempt to dynamically switch to Wav2Vec2 on the active nemo_conformer_python backend
            # Must raise ModelFormatError or DependencyMissingError due to backend mismatch,
            # NOT silently fall back to local model adapter!
            with self.assertRaises((ModelFormatError, DependencyMissingError)):
                manager.reload(
                    model_bucket=self.bucket,
                    model_dir="model/Transcription/wav2vec2_large_960h",
                    task="transcription",
                    input_type="audio",
                )

            # Confirm manager state remains controlled and can reload back to NeMo Conformer
            manager.reload(
                model_bucket=self.bucket,
                model_key="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo",
                task="transcription",
                input_type="audio",
            )
            self.assertTrue(manager.is_ready)
        finally:
            manager.unload()
            manager.stop()

    def test_missing_dependency_guard_for_faster_whisper(self):
        """
        Family 1 (Faster Whisper):
        Verify that attempting to load Faster Whisper against Triton cleanly raises
        DependencyMissingError because ctranslate2 is not installed in the container,
        preventing silent fallback or opaque crashes.
        """
        loader = ModelLoader(storage_config=self.storage_cfg)
        try:
            triton_cfg = TritonConfig(
                endpoint_url=TRITON_TEST_URL,
                model_name="faster_whisper",
                backend="python",
                staging_dir="/tmp/poc_multimodel_staging",
                check_model_ready=False,
            )
            model_uri = f"s3://{self.bucket}/model/Transcription/faster_whisper_large_v3"

            with self.assertRaises(DependencyMissingError) as ctx:
                loader.load_for_triton(
                    model_source=model_uri,
                    config=FrameworkConfig(
                        model_bucket=self.bucket,
                        model_dir="model/Transcription/faster_whisper_large_v3",
                        task="transcription",
                        input_type="audio",
                        storage_config=self.storage_cfg,
                        triton_config=triton_cfg,
                    ),
                )
            self.assertIn("ctranslate2", str(ctx.exception))
        finally:
            loader.close()


if __name__ == "__main__":
    unittest.main()
