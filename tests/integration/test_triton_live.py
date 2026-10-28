"""
Live Integration Tests for NVIDIA Triton Inference Server execution in ML Framework.

Tests the complete end-to-end framework execution workflow against the live Option B
Triton container (triton_conformer_python_poc) running the NeMo Conformer CTC model:
1. Server liveness and readiness probe
2. Model readiness probe for 'nemo_conformer_python'
3. Pipeline orchestration through official public ml.load facade API
4. Real-audio inference execution on test3.wav
5. Output validation and transcription verification
6. Telemetry and execution metrics inspection
"""

import os
import sys
import unittest
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import ml, FrameworkConfig, TritonConfig, ExecutionState
from ml_framework.inference.triton_client import TritonClient
from ml_framework.inference.triton_engine import TritonInferenceEngine

TRITON_TEST_URL = os.environ.get("TRITON_SERVER_URL", "http://localhost:8010")
TEST_MODEL_NAME = "nemo_conformer_python"
TEST_AUDIO_PATH = "/tmp/poc_nemo_audio/test3.wav"


class TestTritonLiveIntegration(unittest.TestCase):
    """Live integration test suite communicating with active Triton container."""

    @classmethod
    def setUpClass(cls):
        cls.client = TritonClient(endpoint_url=TRITON_TEST_URL, timeout_seconds=10.0)
        cls.triton_available = cls.client.is_server_ready()
        if cls.triton_available:
            cls.model_available = cls.client.is_model_ready(TEST_MODEL_NAME)
        else:
            cls.model_available = False

    def setUp(self):
        if not self.triton_available:
            self.skipTest(f"Triton server at {TRITON_TEST_URL} is not available. Skipping live test.")
        if not self.model_available:
            self.skipTest(f"Model '{TEST_MODEL_NAME}' is not READY on Triton server. Skipping live test.")

    def test_live_server_and_model_readiness(self):
        """Verify Triton server and nemo_conformer_python model are READY."""
        self.assertTrue(self.client.is_server_live())
        self.assertTrue(self.client.is_server_ready())
        self.assertTrue(self.client.is_model_ready(TEST_MODEL_NAME))

        metadata = self.client.get_model_metadata(TEST_MODEL_NAME)
        self.assertEqual(metadata["name"], TEST_MODEL_NAME)
        input_names = [inp["name"] for inp in metadata["inputs"]]
        output_names = [out["name"] for out in metadata["outputs"]]
        self.assertIn("AUDIO_SIGNAL", input_names)
        self.assertIn("SAMPLE_RATE", input_names)
        self.assertIn("TRANSCRIPT", output_names)

    def test_end_to_end_audio_transcription_via_ml_load(self):
        """Verify complete pipeline execution: ml.load -> execute(test3.wav) -> verify transcript."""
        if not os.path.isfile(TEST_AUDIO_PATH):
            self.skipTest(f"Test audio file '{TEST_AUDIO_PATH}' not found.")

        cfg = FrameworkConfig(
            model_name=TEST_MODEL_NAME,
            timeout_seconds=60.0,
            triton_config=TritonConfig(
                endpoint_url=TRITON_TEST_URL,
                model_name=TEST_MODEL_NAME,
                check_model_ready=True,
                timeout_seconds=60.0,
            ),
        )

        # 1. Load model via ml facade API
        manager = ml.load(config=cfg)
        self.assertTrue(manager.is_ready)
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertIsInstance(manager.inference_engine, TritonInferenceEngine)

        try:
            # 2. Execute transcription on real audio file
            result = manager.execute(TEST_AUDIO_PATH)

            # 3. Validate execution result
            self.assertTrue(result.success, f"Execution failed: {result.error}")
            self.assertIsNotNone(result.output)
            self.assertIn("transcription", result.output)
            transcript = result.output["transcription"].strip()

            # Verify representative words from test3.wav
            self.assertIn("one two", transcript)
            self.assertIn("three", transcript)
            self.assertIn("four", transcript)

            # 4. Verify telemetry and context metadata
            self.assertGreater(result.duration_ms, 0)
            self.assertEqual(result.state, ExecutionState.READY)
            metrics = manager.metrics
            self.assertEqual(metrics["successful_executions"], 1)

        finally:
            manager.stop()
            self.assertEqual(manager.state, ExecutionState.STOPPED)


if __name__ == "__main__":
    unittest.main()
