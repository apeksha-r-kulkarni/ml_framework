"""
Unit tests for Speech audio decoding, preprocessing, and inference.

Tests cover:
- Audio decoding from WAV bytes, stereo to mono conversion, and resampling
- Rejection of empty, malformed, and non-audio binary payloads
- Processor input preparation and tensor shapes
- Speech-to-text CTC transcription and decoding
- Language identification classification and id2label metadata mapping
- Error propagation for missing dependencies, uninitialized models, and malformed outputs
- Resource cleanup on unload and failure
"""

import os
import sys
import io
import wave
import struct
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig
from ml_framework.models.speech_adapter import SpeechModelAdapter
from ml_framework.input.audio import decode_and_resample_audio, is_valid_audio_payload
from ml_framework.errors import (
    ModelLoadError,
    InferenceError,
    InputProcessingError,
    DependencyMissingError,
)


def create_synthetic_wav_bytes(
    duration_sec: float = 0.5,
    sample_rate: int = 8000,
    channels: int = 1,
    frequency: float = 440.0
) -> bytes:
    """Generates synthetic PCM WAV audio bytes for deterministic unit testing."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        if channels == 1:
            raw_data = tone.tobytes()
        else:
            stereo = np.column_stack([tone, tone])
            raw_data = stereo.tobytes()
        wav.writeframes(raw_data)
    return buf.getvalue()


class TestSpeechInference(unittest.TestCase):
    """Unit test suite for Speech audio preprocessing and model inference."""

    def setUp(self):
        self.wav_8k_mono = create_synthetic_wav_bytes(duration_sec=0.5, sample_rate=8000, channels=1)
        self.wav_8k_stereo = create_synthetic_wav_bytes(duration_sec=0.5, sample_rate=8000, channels=2)

    def test_audio_magic_header_validation(self):
        """Test audio header signature detection for WAV and rejection of arbitrary bytes."""
        valid, fmt = is_valid_audio_payload(self.wav_8k_mono)
        self.assertTrue(valid)
        self.assertEqual(fmt, "wav")

        invalid, _ = is_valid_audio_payload(b"random plain text payload")
        self.assertFalse(invalid)

        empty, _ = is_valid_audio_payload(b"")
        self.assertFalse(empty)

    def test_audio_decoding_and_resampling(self):
        """Test decoding WAV bytes and resampling from 8kHz to target 16kHz float32 waveform."""
        waveform, duration = decode_and_resample_audio(self.wav_8k_mono, target_sampling_rate=16000)
        self.assertIsInstance(waveform, np.ndarray)
        self.assertEqual(waveform.dtype, np.float32)
        self.assertEqual(waveform.ndim, 1)
        # 0.5 sec at 16000 Hz = 8000 samples
        self.assertEqual(len(waveform), 8000)
        self.assertAlmostEqual(duration, 0.5, places=2)

    def test_stereo_to_mono_conversion(self):
        """Test that stereo audio is converted to single-channel 1D waveform."""
        waveform, duration = decode_and_resample_audio(self.wav_8k_stereo, target_sampling_rate=16000)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(len(waveform), 8000)

    def test_empty_and_none_audio_rejection(self):
        """Test that None, empty bytes, and empty strings raise InputProcessingError."""
        with self.assertRaises(InputProcessingError):
            decode_and_resample_audio(None)

        with self.assertRaises(InputProcessingError):
            decode_and_resample_audio(b"")

        with self.assertRaises(InputProcessingError):
            decode_and_resample_audio("   ")

    def test_malformed_non_audio_payload_rejection(self):
        """Test that non-audio binary data raises InputProcessingError with actionable message."""
        arbitrary_bytes = b"This is plain text and definitely not an audio recording."
        with self.assertRaises(InputProcessingError) as cm:
            decode_and_resample_audio(arbitrary_bytes)
        self.assertIn("Unsupported or malformed audio data", str(cm.exception))

    def test_file_path_audio_input(self):
        """Test decoding audio from a temporary file path."""
        with tempfile.NamedTemporaryFile("wb", suffix=".wav", delete=False) as f:
            f.write(self.wav_8k_mono)
            tmp_path = f.name

        try:
            waveform, duration = decode_and_resample_audio(tmp_path, target_sampling_rate=16000)
            self.assertEqual(len(waveform), 8000)
            self.assertAlmostEqual(duration, 0.5, places=2)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unloaded_adapter_predict_error(self):
        """Test that calling predict() on unloaded adapter raises ModelLoadError."""
        adapter = SpeechModelAdapter(model_dir="/tmp/dummy")
        with self.assertRaises(ModelLoadError):
            adapter.predict(self.wav_8k_mono)

    def test_speech_to_text_ctc_inference_mocked(self):
        """Test Wav2Vec2 CTC transcription inference and output formatting with mocks."""
        import torch

        adapter = SpeechModelAdapter(
            model_dir="/tmp/mock_wav2vec2",
            architecture="Wav2Vec2ForCTC",
            model_type="wav2vec2",
        )
        adapter.task = "speech_to_text"
        adapter.sampling_rate = 16000

        # Mock PyTorch model
        mock_model = MagicMock()
        mock_output = MagicMock()
        # Logits of shape (1, 10, 32)
        mock_output.logits = torch.randn(1, 10, 32)
        mock_model.return_value = mock_output
        adapter.model = mock_model

        # Mock Processor
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_values": torch.randn(1, 8000)}
        mock_proc.batch_decode.return_value = ["TEST TRANSCRIPTION"]
        adapter.processor = mock_proc
        adapter._is_loaded = True

        result = adapter.predict(self.wav_8k_mono)

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("task"), "speech_to_text")
        self.assertEqual(result.get("text"), "TEST TRANSCRIPTION")
        self.assertIn("confidence", result)
        self.assertAlmostEqual(result.get("duration_seconds", 0.0), 0.5, places=2)
        mock_model.assert_called_once()
        mock_proc.batch_decode.assert_called_once()

    def test_language_identification_inference_mocked(self):
        """Test MMS language identification classification and label resolution with mocks."""
        import torch

        adapter = SpeechModelAdapter(
            model_dir="/tmp/mock_mms_lid",
            architecture="Wav2Vec2ForSequenceClassification",
            model_type="wav2vec2",
        )
        adapter.task = "language_identification"
        adapter.sampling_rate = 16000

        # Mock Model with 3 classes: 0: "ara", 1: "eng", 2: "spa"
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.logits = torch.tensor([[0.2, 5.0, 0.8]])  # Class 1 ("eng") is highest
        mock_model.return_value = mock_output
        mock_model.config = MagicMock()
        mock_model.config.id2label = {0: "ara", 1: "eng", 2: "spa"}
        adapter.model = mock_model

        # Mock Processor / Feature Extractor
        mock_proc = MagicMock()
        mock_proc.return_value = {"input_values": torch.randn(1, 8000)}
        adapter.processor = mock_proc
        adapter._is_loaded = True

        result = adapter.predict(self.wav_8k_mono)

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("task"), "language_identification")
        self.assertEqual(result.get("language"), "eng")
        self.assertEqual(result.get("class_id"), 1)
        self.assertGreater(result.get("confidence", 0.0), 0.9)
        self.assertTrue(len(result.get("top_predictions", [])) > 0)
        self.assertEqual(result["top_predictions"][0]["language"], "eng")

    def test_malformed_model_output_error(self):
        """Test that model output missing logits raises InferenceError."""
        adapter = SpeechModelAdapter(model_dir="/tmp/mock", architecture="Wav2Vec2ForCTC")
        adapter.task = "speech_to_text"
        adapter.sampling_rate = 16000

        mock_model = MagicMock()
        # Output is None or object without logits
        mock_model.return_value = "not_a_model_output"
        adapter.model = mock_model

        mock_proc = MagicMock()
        mock_proc.return_value = {"input_values": MagicMock(to=MagicMock(return_value="tensor"))}
        adapter.processor = mock_proc
        adapter._is_loaded = True

        with self.assertRaises(InferenceError):
            adapter.predict(self.wav_8k_mono)

    def test_unload_lifecycle(self):
        """Test that unload() sets references to None and resets is_loaded."""
        adapter = SpeechModelAdapter(model_dir="/tmp/mock")
        adapter.model = MagicMock()
        adapter.processor = MagicMock()
        adapter._is_loaded = True

        adapter.unload()
        self.assertIsNone(adapter.model)
        self.assertIsNone(adapter.processor)
        self.assertFalse(adapter.is_loaded)


if __name__ == "__main__":
    unittest.main()
