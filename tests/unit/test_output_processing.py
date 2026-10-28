"""
Unit tests for OutputProcessor (Task 6).
"""

import sys
import unittest
from pathlib import Path
import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.output.processor import OutputProcessor
from ml_framework.results.execution_result import ExecutionContext
from ml_framework.errors.exceptions import OutputProcessingError


class TestOutputProcessor(unittest.TestCase):
    """Test suite for OutputProcessor normalization, serialization, and error handling."""

    def setUp(self):
        self.processor = OutputProcessor()
        self.context = ExecutionContext(correlation_id="test-corr-output")

    def test_none_output_raises_error(self):
        """Test that None raw output raises OutputProcessingError."""
        with self.assertRaises(OutputProcessingError) as ctx:
            self.processor.process(None, self.context)
        self.assertIn("cannot be None", str(ctx.exception))

    def test_backend_error_payload_raises_error(self):
        """Test that error response from backend raises OutputProcessingError."""
        error_payload = {"error": "CUDA out of memory", "success": False}
        with self.assertRaises(OutputProcessingError) as ctx:
            self.processor.process(error_payload, self.context)
        self.assertIn("CUDA out of memory", str(ctx.exception))

    def test_transcription_from_dict_standardization(self):
        """Test standardizing transcription dictionary output."""
        raw = {
            "task": "transcription",
            "transcription": "  hello world from conformer  ",
        }
        res = self.processor.process(raw, self.context)
        self.assertEqual(res["task"], "transcription")
        self.assertEqual(res["text"], "hello world from conformer")
        self.assertEqual(res["transcription"], "hello world from conformer")

    def test_transcription_from_string_standardization(self):
        """Test standardizing raw transcription string when task is transcription."""
        proc = OutputProcessor(task="transcription")
        res = proc.process("  speech to text output  ", self.context)
        self.assertEqual(res["task"], "transcription")
        self.assertEqual(res["text"], "speech to text output")
        self.assertEqual(res["transcription"], "speech to text output")

    def test_transcription_from_list_of_tokens(self):
        """Test standardizing token sequence into full text transcription."""
        raw = {"task": "transcription", "transcript": ["hello", "world"]}
        res = self.processor.process(raw, self.context)
        self.assertEqual(res["text"], "hello world")

    def test_language_detection_standardization(self):
        """Test standardizing language detection output."""
        raw = {
            "task": "language_detection",
            "language": "en",
            "confidence": 0.985,
        }
        res = self.processor.process(raw, self.context)
        self.assertEqual(res["task"], "language_detection")
        self.assertEqual(res["language"], "en")
        self.assertEqual(res["label"], "en")
        self.assertAlmostEqual(res["confidence"], 0.985)

    def test_denoising_standardization(self):
        """Test standardizing audio denoising output."""
        clean_samples = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        raw = {
            "task": "denoising",
            "enhanced_audio": clean_samples,
            "sample_rate": 16000,
        }
        res = self.processor.process(raw, self.context)
        self.assertEqual(res["task"], "denoising")
        self.assertEqual(res["sample_rate"], 16000)
        np.testing.assert_array_almost_equal(res["enhanced_audio"], clean_samples)

    def test_custom_formatter_override(self):
        """Test registering a custom formatter callback."""
        def custom_fmt(raw, ctx):
            return {"custom_key": str(raw).upper(), "corr": ctx.correlation_id}

        proc = OutputProcessor(task="custom_task", custom_formatters={"custom_task": custom_fmt})
        res = proc.process("sample data", self.context)
        self.assertEqual(res["custom_key"], "SAMPLE DATA")
        self.assertEqual(res["corr"], "test-corr-output")

    def test_serialization_converts_numpy_arrays_to_list(self):
        """Test that serialize converts NumPy arrays and types to JSON-safe primitives."""
        payload = {
            "waveform": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "score": np.float32(0.95),
            "index": np.int64(42),
            "flag": np.bool_(True),
        }
        serialized = OutputProcessor.serialize(payload)
        self.assertIsInstance(serialized["waveform"], list)
        self.assertIsInstance(serialized["score"], float)
        self.assertIsInstance(serialized["index"], int)
        self.assertIsInstance(serialized["flag"], bool)

        # Ensure valid JSON
        import json
        json_str = OutputProcessor.to_json(payload)
        parsed = json.loads(json_str)
        self.assertAlmostEqual(parsed["score"], 0.95, places=5)
        self.assertEqual(parsed["index"], 42)
        self.assertTrue(parsed["flag"])


if __name__ == "__main__":
    unittest.main()
