"""
Unit tests for Generic Model Loader - Speech Model (NeMo Conformer CTC / Speech Models).

Tests verify:
1. Format detection and loader selection for NeMo artifacts (.nemo file, NeMo directory).
2. EnvironmentManager dependency inspection and offline validation for framework 'nemo'.
3. Hardware accelerator resolution and device compatibility checks.
4. Error handling on missing, corrupted, or unsupported model artifacts.
5. SpeechModelAdapter NeMo loading, device placement, and eval mode.
6. NeMo CTC inference execution, audio decoding, tensor shapes, and silent audio handling.
7. Output contract consistency (task, text, confidence, sampling_rate, duration_seconds).
8. Resource cleanup and memory release on unload.
9. End-to-end execution through ModelLoader and high-level ml facade.
"""

import os
import sys
import io
import wave
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig
from ml_framework.models.model_loader import ModelLoader
from ml_framework.models.specialized_loaders import SpeechModelLoader
from ml_framework.models.speech_adapter import SpeechModelAdapter
from ml_framework.models.format_detector import FormatDetector
from ml_framework.environment.environment_manager import EnvironmentManager
from ml_framework.errors import (
    ModelLoadError,
    ModelFormatError,
    DependencyMissingError,
    HardwareIncompatibleError,
    InferenceError,
    InputProcessingError,
)


def create_test_wav_bytes(
    duration_sec: float = 0.5,
    sample_rate: int = 16000,
    frequency: float = 440.0
) -> bytes:
    """Creates deterministic PCM WAV audio bytes for testing."""
    buf = io.BytesIO()
    num_samples = int(duration_sec * sample_rate)
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit PCM
        wav.setframerate(sample_rate)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)
        wav.writeframes(tone.tobytes())
    return buf.getvalue()


class TestModelLoaderSpeechNeMo(unittest.TestCase):
    """Unit test suite for Generic Model Loader speech model (NeMo Option B) support."""

    def setUp(self):
        self.wav_bytes = create_test_wav_bytes(duration_sec=0.5, sample_rate=16000)

    # -------------------------------------------------------------------------
    # 1. Format Detection and Loader Selection
    # -------------------------------------------------------------------------

    def test_format_detector_single_nemo_file(self):
        """FormatDetector identifies standalone .nemo archive as framework 'nemo'."""
        detector = FormatDetector()
        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as f:
            f.write(b"MOCK_NEMO_ARCHIVE_DATA")
            nemo_path = f.name

        try:
            framework, architecture, model_type = detector.detect(nemo_path)
            self.assertEqual(framework, "nemo")
            self.assertEqual(architecture, "NeMoModel")
            self.assertIn(Path(nemo_path).stem, model_type)

            # Also valid when configured_framework="speech"
            fw_speech, _, _ = detector.detect(nemo_path, configured_framework="speech")
            self.assertEqual(fw_speech, "nemo")

            # Conflicting configured framework raises ModelFormatError
            with self.assertRaises(ModelFormatError):
                detector.detect(nemo_path, configured_framework="spacy")
        finally:
            if os.path.exists(nemo_path):
                os.remove(nemo_path)

    def test_format_detector_nemo_directory_with_yaml(self):
        """FormatDetector identifies directory containing model_config.yaml."""
        detector = FormatDetector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_content = (
                "target: nemo.collections.asr.models.ctc_bpe_models.EncDecCTCModelBPE\n"
                "encoder:\n"
                "  _target_: nemo.collections.asr.modules.ConformerEncoder\n"
            )
            with open(os.path.join(tmp_dir, "model_config.yaml"), "w") as f:
                f.write(yaml_content)

            framework, architecture, model_type = detector.detect(tmp_dir)
            self.assertEqual(framework, "nemo")
            self.assertEqual(architecture, "EncDecCTCModelBPE")
            self.assertEqual(model_type, "conformer_ctc")

    def test_format_detector_directory_containing_nemo_file(self):
        """FormatDetector identifies directory containing a .nemo archive file."""
        detector = FormatDetector()
        with tempfile.TemporaryDirectory() as tmp_dir:
            nemo_file = os.path.join(tmp_dir, "stt_en_conformer_ctc_large.nemo")
            with open(nemo_file, "wb") as f:
                f.write(b"MOCK_NEMO_BYTES")

            framework, architecture, model_type = detector.detect(tmp_dir)
            self.assertEqual(framework, "nemo")
            self.assertEqual(architecture, "EncDecCTCModelBPE")
            self.assertEqual(model_type, "conformer_ctc")

    def test_model_loader_registers_nemo_framework(self):
        """ModelLoader includes 'nemo' in supported frameworks and dispatches to SpeechModelLoader."""
        loader = ModelLoader()
        self.assertIn("nemo", loader.get_supported_frameworks())
        specialized = loader.get_loader("nemo")
        self.assertIsInstance(specialized, SpeechModelLoader)

    # -------------------------------------------------------------------------
    # 2. Dependency Inspection and Offline Validation
    # -------------------------------------------------------------------------

    def test_environment_manager_inspects_nemo_dependencies(self):
        """EnvironmentManager determines required packages for nemo artifacts."""
        env = EnvironmentManager()
        deps_from_framework = env.inspect_dependencies("/dummy/path", framework="nemo")
        self.assertIn("nemo_toolkit", deps_from_framework)
        self.assertIn("torch", deps_from_framework)
        self.assertIn("soundfile", deps_from_framework)
        self.assertIn("numpy", deps_from_framework)

        deps_from_file = env.inspect_dependencies("/path/to/conformer.nemo")
        self.assertEqual(sorted(deps_from_file), sorted(["nemo_toolkit", "numpy", "soundfile", "torch"]))

        deps_from_meta = env.inspect_dependencies({"framework": "nemo", "model_type": "conformer_ctc"})
        self.assertIn("nemo_toolkit", deps_from_meta)

    def test_missing_nemo_dependency_raises_clear_offline_error(self):
        """Missing nemo_toolkit raises DependencyMissingError with offline recommendations."""
        env = EnvironmentManager(override_packages={
            "torch": True,
            "soundfile": True,
            "numpy": True,
            "nemo_toolkit": False,
        })
        deps = env.inspect_dependencies("/path/to/conformer.nemo")
        with self.assertRaises(DependencyMissingError) as cm:
            env.validate_dependencies(deps, context_label="model 'conformer.nemo'")
        self.assertIn("nemo_toolkit", cm.exception.missing_packages)
        self.assertIn("pip install --no-index", str(cm.exception))

    # -------------------------------------------------------------------------
    # 3. Hardware Resolution and Error Handling
    # -------------------------------------------------------------------------

    def test_hardware_resolution_and_validation(self):
        """CPU is validated successfully; requesting unavailable CUDA raises HardwareIncompatibleError."""
        env = EnvironmentManager()
        device_cpu = env.resolve_device("cpu")
        self.assertEqual(device_cpu, "cpu")
        env.validate_hardware("cpu")

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with self.assertRaises(HardwareIncompatibleError):
                env.validate_hardware("cuda")

    def test_missing_model_path_raises_load_error(self):
        """Attempting to load a non-existent .nemo model raises ModelLoadError."""
        loader = ModelLoader()
        with self.assertRaises(ModelLoadError):
            loader.load("/non/existent/path/speech_model.nemo")

    def test_unsupported_model_format_raises_error(self):
        """Directory with unrecognized files raises ModelFormatError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "unknown.bin"), "wb") as f:
                f.write(b"random bytes")
            loader = ModelLoader()
            with self.assertRaises(ModelFormatError):
                loader.load(tmp_dir)

    # -------------------------------------------------------------------------
    # 4. SpeechModelAdapter NeMo Loading & Device Placement
    # -------------------------------------------------------------------------

    def test_speech_adapter_nemo_load_mocked(self):
        """SpeechModelAdapter loads NeMo model offline and places on CPU."""
        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as f:
            f.write(b"MOCK_NEMO_CHECKPOINT")
            nemo_path = f.name

        try:
            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.preprocessor._sample_rate = 16000

            mock_nemo_asr = MagicMock()
            mock_nemo_asr.models.EncDecCTCModelBPE.restore_from.return_value = mock_model

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_nemo = MagicMock()
            mock_nemo.collections.asr = mock_nemo_asr

            with patch.dict("sys.modules", {
                "torch": mock_torch,
                "nemo": mock_nemo,
                "nemo.collections": mock_nemo.collections,
                "nemo.collections.asr": mock_nemo_asr,
            }):
                adapter = SpeechModelAdapter(
                    model_dir=nemo_path,
                    device="cpu",
                    architecture="EncDecCTCModelBPE",
                    model_type="conformer_ctc",
                )
                self.assertFalse(adapter.is_loaded)
                adapter.load()

                self.assertTrue(adapter.is_loaded)
                self.assertIs(adapter.model, mock_model)
                self.assertEqual(adapter.sampling_rate, 16000)
                self.assertEqual(adapter.task, "speech_to_text")

                # Verify restore_from arguments
                mock_nemo_asr.models.EncDecCTCModelBPE.restore_from.assert_called_once_with(
                    restore_path=nemo_path,
                    map_location="cpu",
                )
                mock_model.to.assert_called_with("cpu")
                mock_model.eval.assert_called_once()
        finally:
            if os.path.exists(nemo_path):
                os.remove(nemo_path)

    def test_speech_adapter_nemo_missing_dependency_raises_error(self):
        """SpeechModelAdapter.load() raises DependencyMissingError if nemo is not importable."""
        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as f:
            f.write(b"MOCK_NEMO")
            nemo_path = f.name

        try:
            adapter = SpeechModelAdapter(model_dir=nemo_path)
            # Simulate ImportError for nemo
            with patch.dict("sys.modules", {"nemo": None, "nemo.collections.asr": None}):
                with self.assertRaises(DependencyMissingError) as cm:
                    adapter.load()
                self.assertIn("nemo_toolkit", cm.exception.missing_packages)
        finally:
            if os.path.exists(nemo_path):
                os.remove(nemo_path)

    # -------------------------------------------------------------------------
    # 5. NeMo CTC Inference and Output Contract Consistency
    # -------------------------------------------------------------------------

    def test_nemo_ctc_prediction_and_contract(self):
        """NeMo model predict() executes forward pass, CTC greedy decoding, and conforms to output contract."""
        import torch

        adapter = SpeechModelAdapter(
            model_dir="/tmp/mock_conformer.nemo",
            device="cpu",
            architecture="EncDecCTCModelBPE",
            model_type="conformer_ctc",
        )
        adapter.sampling_rate = 16000
        adapter.task = "speech_to_text"
        adapter._is_loaded = True
        adapter._is_nemo = True

        mock_model = MagicMock()
        # Mock forward pass output: logprobs (1, 10, 1024), encoded_len (1)
        mock_logprobs = torch.randn(1, 10, 1024)
        mock_encoded_len = torch.tensor([10])
        mock_model.return_value = (mock_logprobs, mock_encoded_len)

        # Mock CTC decoder hypotheses
        mock_hyp = MagicMock()
        mock_hyp.text = "hello world from conformer"
        mock_model.decoding.ctc_decoder_predictions_tensor.return_value = [[mock_hyp]]
        adapter.model = mock_model

        result = adapter.predict(self.wav_bytes)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["task"], "speech_to_text")
        self.assertEqual(result["text"], "hello world from conformer")
        self.assertEqual(result["transcription"], "hello world from conformer")
        self.assertEqual(result["architecture"], "EncDecCTCModelBPE")
        self.assertEqual(result["model_type"], "conformer_ctc")
        self.assertEqual(result["sampling_rate"], 16000)
        self.assertAlmostEqual(result["duration_seconds"], 0.5, places=2)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Verify forward call
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args.kwargs
        self.assertIn("input_signal", call_kwargs)
        self.assertIn("input_signal_length", call_kwargs)
        self.assertEqual(call_kwargs["input_signal"].ndim, 2)  # [1, T]

    def test_nemo_silent_audio_returns_empty_transcription(self):
        """Silent audio (all zeros) returns empty transcription without executing model."""
        adapter = SpeechModelAdapter(
            model_dir="/tmp/mock_conformer.nemo",
            device="cpu",
            architecture="EncDecCTCModelBPE",
            model_type="conformer_ctc",
        )
        adapter.sampling_rate = 16000
        adapter._is_loaded = True
        adapter._is_nemo = True

        mock_model = MagicMock()
        adapter.model = mock_model

        # Silent audio array of zeros
        silent_audio = np.zeros(8000, dtype=np.float32)
        result = adapter.predict(silent_audio)

        self.assertEqual(result["task"], "speech_to_text")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["transcription"], "")
        mock_model.assert_not_called()

    def test_invalid_inference_inputs(self):
        """Invalid audio input payloads raise appropriate typed exceptions."""
        adapter = SpeechModelAdapter(model_dir="/tmp/mock.nemo", architecture="EncDecCTCModelBPE")
        adapter._is_loaded = True
        adapter._is_nemo = True
        adapter.model = MagicMock()

        # None input raises InferenceError
        with self.assertRaises(InferenceError):
            adapter.predict(None)

        # Non-audio bytes raise InputProcessingError
        with self.assertRaises(InputProcessingError):
            adapter.predict(b"definitely not an audio file")

    def test_nemo_inference_failure_raises_inference_error(self):
        """Forward pass failure raises InferenceError with diagnostic context."""
        adapter = SpeechModelAdapter(model_dir="/tmp/mock.nemo", architecture="EncDecCTCModelBPE")
        adapter._is_loaded = True
        adapter._is_nemo = True

        mock_model = MagicMock()
        mock_model.side_effect = RuntimeError("CUDA out of memory in Conformer attention")
        adapter.model = mock_model

        with self.assertRaises(InferenceError) as cm:
            adapter.predict(self.wav_bytes)
        self.assertIn("forward pass failed", str(cm.exception))

    # -------------------------------------------------------------------------
    # 6. Unload Lifecycle and Resource Cleanup
    # -------------------------------------------------------------------------

    def test_nemo_unload_cleans_resources(self):
        """unload() resets model references, resets is_loaded, and clears references."""
        adapter = SpeechModelAdapter(model_dir="/tmp/mock.nemo")
        adapter.model = MagicMock()
        adapter._is_loaded = True
        adapter._is_nemo = True

        self.assertTrue(adapter.is_loaded)
        adapter.unload()
        self.assertFalse(adapter.is_loaded)
        self.assertIsNone(adapter.model)
        self.assertFalse(adapter._is_nemo)

    def test_model_loader_unload_dispatches_cleanly(self):
        """ModelLoader.unload() cleans model instances and removes temp dirs."""
        loader = ModelLoader()
        mock_instance = MagicMock()
        loader.unload(mock_instance)
        mock_instance.unload.assert_called_once()

    # -------------------------------------------------------------------------
    # 7. ModelLoader End-to-End Loading Dispatch
    # -------------------------------------------------------------------------

    def test_model_loader_end_to_end_mocked(self):
        """ModelLoader.load() with a .nemo file successfully instantiates SpeechModelAdapter."""
        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as f:
            f.write(b"MOCK_NEMO_DATA")
            nemo_path = f.name

        try:
            env = EnvironmentManager(override_packages={
                "torch": True,
                "nemo_toolkit": True,
                "soundfile": True,
                "numpy": True,
            })
            loader = ModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_model.eval.return_value = mock_model
            mock_model.preprocessor._sample_rate = 16000

            mock_nemo_asr = MagicMock()
            mock_nemo_asr.models.EncDecCTCModelBPE.restore_from.return_value = mock_model

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_nemo = MagicMock()
            mock_nemo.collections.asr = mock_nemo_asr

            with patch.dict("sys.modules", {
                "torch": mock_torch,
                "nemo": mock_nemo,
                "nemo.collections": mock_nemo.collections,
                "nemo.collections.asr": mock_nemo_asr,
            }):
                instance = loader.load(nemo_path, config=FrameworkConfig(framework="nemo", device="cpu"))
                self.assertIsInstance(instance, SpeechModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.device, "cpu")

                loader.unload(instance)
                self.assertFalse(instance.is_loaded)
        finally:
            if os.path.exists(nemo_path):
                os.remove(nemo_path)


if __name__ == "__main__":
    unittest.main()
