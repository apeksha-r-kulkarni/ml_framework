"""
Unit tests for Generic Model Loader POC — Multi-Loader and Framework Adapter Subsystem.

Tests format detection, generic loader selection, specialized loader dispatch,
offline loading constraints, environment validation, hardware checks,
error handling, and model unloading across Speech, spaCy, Transformers, and ONNX.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import ml, FrameworkConfig, StorageConfig, ExecutionState
from ml_framework.environment import EnvironmentManager
from ml_framework.interfaces import BaseModelLoader, BaseModelAdapter
from ml_framework.models import (
    ModelLoader,
    SpeechModelLoader,
    SpacyModelLoader,
    TransformersModelLoader,
    ONNXModelLoader,
    SpeechModelAdapter,
    SpacyModelAdapter,
    TransformersModelAdapter,
    ONNXModelAdapter,
    FormatDetector,
)
from ml_framework.errors import (
    ModelLoadError,
    ModelFormatError,
    DependencyMissingError,
    HardwareIncompatibleError,
    InferenceError,
)


class TestFormatDetector(unittest.TestCase):
    """Test suite for FormatDetector component across all supported formats."""

    def setUp(self):
        self.detector = FormatDetector()

    def test_detect_whisper_speech_directory(self):
        """Test detecting a Whisper speech model directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"]
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "speech")
            self.assertEqual(arch, "WhisperForConditionalGeneration")
            self.assertEqual(m_type, "whisper")

    def test_detect_wav2vec2_speech_directory(self):
        """Test detecting a Wav2Vec2 CTC speech model directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "wav2vec2",
                "architectures": ["Wav2Vec2ForCTC"]
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "speech")
            self.assertEqual(arch, "Wav2Vec2ForCTC")
            self.assertEqual(m_type, "wav2vec2")

    def test_detect_speech_via_preprocessor_config(self):
        """Test detecting speech framework via preprocessor_config.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "custom_audio_net", "architectures": ["CustomNet"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            pre = {"feature_extractor_type": "WhisperFeatureExtractor"}
            with open(os.path.join(tmp_dir, "preprocessor_config.json"), "w") as f:
                json.dump(pre, f)

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "speech")

    def test_detect_standalone_onnx_file(self):
        """Test detecting standalone .onnx file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            framework, arch, m_type = self.detector.detect(tmp_file.name)
            self.assertEqual(framework, "onnx")
            self.assertEqual(arch, "ONNXModel")

    def test_detect_spacy_directory(self):
        """Test detecting spaCy model directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump({"name": "en_core_web_sm"}, f)

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "spacy")
            self.assertEqual(arch, "spaCyPipeline")
            self.assertEqual(m_type, "en_core_web_sm")

    def test_detect_transformers_text_directory(self):
        """Test detecting non-speech HuggingFace text model directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "bert",
                "architectures": ["BertForSequenceClassification"]
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "transformers")
            self.assertEqual(arch, "BertForSequenceClassification")
            self.assertEqual(m_type, "bert")

    def test_empty_directory_raises_format_error(self):
        """Test that an empty directory raises ModelFormatError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ModelFormatError) as cm:
                self.detector.detect(tmp_dir)
            self.assertIn("is empty", str(cm.exception))

    def test_missing_path_raises_load_error(self):
        """Test that non-existent path raises ModelLoadError."""
        with self.assertRaises(ModelLoadError) as cm:
            self.detector.detect("/non/existent/model_dir_xyz")
        self.assertIn("does not exist", str(cm.exception))

    def test_corrupted_config_json_raises_format_error(self):
        """Test that invalid config.json raises ModelFormatError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                f.write("not valid json {[[[")
            with self.assertRaises(ModelFormatError) as cm:
                self.detector.detect(tmp_dir)
            self.assertIn("Corrupted or invalid config.json", str(cm.exception))

    def test_configured_speech_framework_mismatch(self):
        """Test configuring framework='speech' for a BERT text model raises ModelFormatError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with self.assertRaises(ModelFormatError) as cm:
                self.detector.detect(tmp_dir, configured_framework="speech")
            self.assertIn("Configured framework is 'speech', but model", str(cm.exception))

    def test_configured_spacy_framework_mismatch(self):
        """Test configuring framework='spacy' for a Transformers model raises ModelFormatError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with self.assertRaises(ModelFormatError) as cm:
                self.detector.detect(tmp_dir, configured_framework="spacy")
            self.assertIn("Configured framework is 'spacy'", str(cm.exception))

    def test_detect_architectures_null_handled_safely(self):
        """Test that config.json with 'architectures': null does not raise TypeError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump({"model_type": "whisper", "architectures": None}, f)
            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "speech")
            self.assertEqual(m_type, "whisper")
            self.assertEqual(arch, "Model")

    def test_detect_spacy_directory_with_meta_and_config_cfg(self):
        """Test that a valid spaCy directory containing meta.json and config.cfg is detected as spaCy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = {
                "lang": "en",
                "name": "core_web_trf",
                "version": "3.8.0",
                "spacy_version": ">=3.8.0,<3.9.0",
                "pipeline": ["transformer", "tagger", "parser", "ner"],
            }
            with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[nlp]\nlang = \"en\"\npipeline = [\"transformer\",\"tagger\",\"parser\",\"ner\"]\n")

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "spacy")
            self.assertEqual(arch, "spaCyPipeline")
            self.assertEqual(m_type, "core_web_trf")

    def test_detect_transformers_directory_with_config_cfg_and_safetensors(self):
        """Test that a Transformers directory containing config.json, config.cfg, and model.safetensors is detected as transformers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "distilbert",
                "architectures": ["DistilBertForTokenClassification"],
                "transformers_version": "4.52.1",
            }
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[unrelated]\nkey=value\n")
            with open(os.path.join(tmp_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00\x00\x00\x00dummy_safetensors_data")

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "transformers")
            self.assertEqual(arch, "DistilBertForTokenClassification")
            self.assertEqual(m_type, "distilbert")

    def test_detect_standard_transformers_directory_with_weights_no_config_cfg(self):
        """Test that a standard Transformers directory containing config.json and weights without config.cfg is detected as transformers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "roberta",
                "architectures": ["RobertaForSequenceClassification"],
                "transformers_version": "4.40.0",
            }
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with open(os.path.join(tmp_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00\x00\x00\x00dummy_safetensors_data")

            framework, arch, m_type = self.detector.detect(tmp_dir)
            self.assertEqual(framework, "transformers")
            self.assertEqual(arch, "RobertaForSequenceClassification")
            self.assertEqual(m_type, "roberta")

    def test_transformers_with_config_cfg_explicit_framework_and_conflicts(self):
        """Test explicit framework selection and conflicting configurations on Transformers directory with config.cfg."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "distilbert",
                "architectures": ["DistilBertForTokenClassification"],
            }
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[unrelated]\nkey=value\n")
            with open(os.path.join(tmp_dir, "model.safetensors"), "wb") as f:
                f.write(b"\x00\x00\x00\x00dummy_safetensors_data")

            # Explicit compatible framework selections
            fw, arch, m_type = self.detector.detect(tmp_dir, configured_framework="transformers")
            self.assertEqual(fw, "transformers")
            self.assertEqual(arch, "DistilBertForTokenClassification")

            fw, arch, m_type = self.detector.detect(tmp_dir, configured_framework="huggingface")
            self.assertEqual(fw, "transformers")
            self.assertEqual(arch, "DistilBertForTokenClassification")

            # Conflicting framework: spacy on Transformers directory with config.cfg
            with self.assertRaises(ModelFormatError) as cm_spacy:
                self.detector.detect(tmp_dir, configured_framework="spacy")
            self.assertIn("is a Transformers model", str(cm_spacy.exception))

            # Conflicting framework: speech on non-speech Transformers directory with config.cfg
            with self.assertRaises(ModelFormatError) as cm_speech:
                self.detector.detect(tmp_dir, configured_framework="speech")
            self.assertIn("Configured framework is 'speech', but model", str(cm_speech.exception))

            # Conflicting framework: onnx on Transformers directory without ONNX files
            with self.assertRaises(ModelFormatError) as cm_onnx:
                self.detector.detect(tmp_dir, configured_framework="onnx")
            self.assertIn("is a Transformers model", str(cm_onnx.exception))

    def test_spacy_directory_with_conflicting_framework_selection(self):
        """Test conflicting framework selections on a valid spaCy directory containing meta.json and config.cfg."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = {"name": "core_web_trf", "spacy_version": ">=3.8.0"}
            with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[nlp]\nlang = \"en\"\n")

            # Explicit compatible framework selection
            fw, arch, m_type = self.detector.detect(tmp_dir, configured_framework="spacy")
            self.assertEqual(fw, "spacy")
            self.assertEqual(m_type, "core_web_trf")

            # Conflicting framework selections
            with self.assertRaises(ModelFormatError) as cm_tf:
                self.detector.detect(tmp_dir, configured_framework="transformers")
            self.assertIn("is a spaCy pipeline", str(cm_tf.exception))

            with self.assertRaises(ModelFormatError) as cm_speech:
                self.detector.detect(tmp_dir, configured_framework="speech")
            self.assertIn("is a spaCy pipeline", str(cm_speech.exception))

    def test_corrupted_config_json_with_config_cfg_present_raises_format_error(self):
        """Test that invalid config.json in the presence of config.cfg raises ModelFormatError and is not misclassified as spaCy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                f.write("{invalid json content missing braces")
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[unrelated]\nkey=value\n")

            with self.assertRaises(ModelFormatError) as cm:
                self.detector.detect(tmp_dir)
            self.assertIn("Corrupted or invalid config.json", str(cm.exception))


class TestSpeechModelLoaderAndAdapter(unittest.TestCase):
    """Test suite for SpeechModelLoader and SpeechModelAdapter."""

    def _create_mock_transformers(self, hf_config, model=None, processor=None, model_error=None):
        mock_tf = MagicMock()
        mock_tf.AutoConfig.from_pretrained.return_value = hf_config
        if model_error:
            mock_tf.AutoModelForSpeechSeq2Seq.from_pretrained.side_effect = model_error
            mock_tf.AutoModelForCTC.from_pretrained.side_effect = model_error
            mock_tf.AutoModel.from_pretrained.side_effect = model_error
        else:
            mock_tf.AutoModelForSpeechSeq2Seq.from_pretrained.return_value = model
            mock_tf.AutoModelForCTC.from_pretrained.return_value = model
            mock_tf.AutoModel.from_pretrained.return_value = model
        if processor:
            mock_tf.AutoProcessor.from_pretrained.return_value = processor
        else:
            mock_tf.AutoProcessor.from_pretrained.side_effect = Exception("No processor")
        return mock_tf

    def test_speech_model_missing_dependency_raises_clear_error(self):
        """Test that missing speech dependencies trigger DependencyMissingError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "soundfile": False,
                "numpy": True,
            })
            loader = ModelLoader(env_manager=env)

            with self.assertRaises(DependencyMissingError) as cm:
                loader.load(tmp_dir)
            self.assertIn("soundfile", cm.exception.missing_packages)
            self.assertIn("Offline installation recommendation", str(cm.exception))

    def test_speech_model_incompatible_hardware_raises_error(self):
        """Test that requesting unavailable CUDA device raises HardwareIncompatibleError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "soundfile": True,
                "numpy": True,
            })
            loader = ModelLoader(env_manager=env)
            framework_cfg = FrameworkConfig(device="cuda")

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with self.assertRaises(HardwareIncompatibleError) as cm:
                    loader.load(tmp_dir, config=framework_cfg)
                self.assertIn("no CUDA-compatible GPU is available", str(cm.exception))

    def test_speech_adapter_load_and_unload_mocked(self):
        """Test SpeechModelAdapter lifecycle with mocked Transformers classes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
                "is_encoder_decoder": True,
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.is_encoder_decoder = True
            mock_hf_config.model_type = "whisper"
            mock_hf_config.architectures = ["WhisperForConditionalGeneration"]

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_tf = self._create_mock_transformers(mock_hf_config, mock_model, mock_processor)

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = SpeechModelAdapter(
                    model_dir=tmp_dir,
                    device="cpu",
                    architecture="WhisperForConditionalGeneration",
                    model_type="whisper",
                )
                self.assertFalse(adapter.is_loaded)

                adapter.load()
                self.assertTrue(adapter.is_loaded)
                self.assertIs(adapter.model, mock_model)
                self.assertIs(adapter.processor, mock_processor)

                # Verify local_files_only=True was passed
                mock_tf.AutoConfig.from_pretrained.assert_called_with(tmp_dir, local_files_only=True)
                mock_tf.AutoModelForSpeechSeq2Seq.from_pretrained.assert_called_with(
                    tmp_dir, config=mock_hf_config, local_files_only=True
                )
                mock_tf.AutoProcessor.from_pretrained.assert_called_with(tmp_dir, local_files_only=True)

                mock_model.to.assert_called_with("cpu")
                mock_processor.return_value = {"input_features": MagicMock()}
                mock_processor.batch_decode.return_value = ["mocked speech transcription"]
                mock_model.generate.return_value = MagicMock()

                audio_arr = np.zeros(16000, dtype=np.float32)
                res = adapter.predict(audio_arr)
                self.assertIsInstance(res, dict)
                self.assertEqual(res["task"], "speech_seq2seq")
                self.assertEqual(res["transcription"], "mocked speech transcription")

                adapter.unload()
                self.assertFalse(adapter.is_loaded)
                self.assertIsNone(adapter.model)
                self.assertIsNone(adapter.processor)

    def test_model_loader_speech_poc_end_to_end_mocked(self):
        """Test ModelLoader.load() returns loaded SpeechModelAdapter and unloads cleanly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"],
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "soundfile": True,
                "numpy": True,
            })
            loader = ModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.is_encoder_decoder = True
            mock_hf_config.model_type = "whisper"

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_tf = self._create_mock_transformers(mock_hf_config, mock_model, mock_processor)

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                instance = loader.load(tmp_dir, config=FrameworkConfig(framework="speech", device="cpu"))
                self.assertIsInstance(instance, SpeechModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.architecture, "WhisperForConditionalGeneration")
                self.assertEqual(instance.device, "cpu")

                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_speech_model_weight_load_failure_raises_model_load_error(self):
        """Test that weight loading failure raises ModelLoadError with actionable context."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "soundfile": True,
                "numpy": True,
            })
            loader = ModelLoader(env_manager=env)

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_tf = self._create_mock_transformers(
                MagicMock(), model_error=OSError("No weights found in directory")
            )

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                with self.assertRaises(ModelLoadError) as cm:
                    loader.load(tmp_dir)
                self.assertIn("Failed to load speech model weights", str(cm.exception))

    def test_ml_load_facade_integration_with_speech_model(self):
        """Test full ml.load() integration path loading a speech model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "wav2vec2",
                "architectures": ["Wav2Vec2ForCTC"],
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env_mgr = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "soundfile": True,
                "numpy": True,
            })

            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.is_encoder_decoder = False
            mock_hf_config.model_type = "wav2vec2"

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_tf = self._create_mock_transformers(mock_hf_config, mock_model, mock_processor)

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                manager = ml.load(
                    model_source=tmp_dir,
                    framework="speech",
                    device="cpu",
                    env_manager=env_mgr,
                )

                self.assertTrue(manager.is_ready)
                self.assertEqual(manager.state, ExecutionState.READY)
                self.assertIsInstance(manager._active_model_instance, SpeechModelAdapter)
                self.assertTrue(manager._active_model_instance.is_loaded)

                manager.unload()
                self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
                manager.stop()
                self.assertEqual(manager.state, ExecutionState.STOPPED)

    def test_speech_specialized_loader_standalone(self):
        """Test SpeechModelLoader directly as a standalone specialized loader."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={
                "torch": True, "transformers": True, "soundfile": True, "numpy": True
            })
            loader = SpeechModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_processor = MagicMock()
            mock_hf_config = MagicMock(is_encoder_decoder=True, model_type="whisper")
            mock_tf = self._create_mock_transformers(mock_hf_config, mock_model, mock_processor)
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = loader.load(tmp_dir)
                self.assertIsInstance(adapter, SpeechModelAdapter)
                self.assertTrue(adapter.is_loaded)
                loader.unload(adapter)
                self.assertFalse(adapter.is_loaded)


class TestSpacyModelLoaderAndAdapter(unittest.TestCase):
    """Test suite for SpacyModelLoader and SpacyModelAdapter."""

    def test_spacy_adapter_load_and_unload_mocked(self):
        """Test SpacyModelAdapter lifecycle with mocked spaCy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump({"name": "en_core_web_sm", "version": "3.7.0"}, f)

            mock_nlp = MagicMock()
            mock_nlp.meta = {"name": "en_core_web_sm"}
            mock_nlp.return_value = "parsed_doc"

            mock_spacy = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            with patch.dict("sys.modules", {"spacy": mock_spacy}):
                adapter = SpacyModelAdapter(
                    model_dir=tmp_dir,
                    device="cpu",
                    architecture="spaCyPipeline",
                    model_type="en_core_web_sm",
                )
                self.assertFalse(adapter.is_loaded)

                adapter.load()
                self.assertTrue(adapter.is_loaded)
                mock_spacy.load.assert_called_with(tmp_dir)

                res = adapter.predict("Sample text to parse")
                self.assertEqual(res, "parsed_doc")
                mock_nlp.assert_called_with("Sample text to parse")

                adapter.unload()
                self.assertFalse(adapter.is_loaded)
                self.assertIsNone(adapter.nlp)

    def test_spacy_adapter_missing_dependency(self):
        """Test SpacyModelAdapter raises DependencyMissingError when spaCy is not installed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = SpacyModelAdapter(model_dir=tmp_dir)
            with patch.dict("sys.modules", {"spacy": None}):
                with self.assertRaises(DependencyMissingError) as cm:
                    adapter.load()
                self.assertIn("spacy", cm.exception.missing_packages)
                self.assertIn("pip install", str(cm.exception))

    def test_spacy_adapter_hardware_incompatible_gpu(self):
        """Test requesting GPU when prefer_gpu() returns False raises HardwareIncompatibleError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_spacy = MagicMock()
            mock_spacy.prefer_gpu.return_value = False

            with patch.dict("sys.modules", {"spacy": mock_spacy}):
                adapter = SpacyModelAdapter(model_dir=tmp_dir, device="cuda")
                with self.assertRaises(HardwareIncompatibleError) as cm:
                    adapter.load()
                self.assertIn("GPU device 'cuda' requested for spaCy", str(cm.exception))

    def test_spacy_loader_standalone(self):
        """Test SpacyModelLoader directly as a standalone specialized loader."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump({"name": "en_core_web_sm"}, f)

            env = EnvironmentManager(override_packages={"spacy": True})
            loader = SpacyModelLoader(env_manager=env)

            mock_nlp = MagicMock()
            mock_nlp.meta = {"name": "en_core_web_sm"}
            mock_spacy = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            with patch.dict("sys.modules", {"spacy": mock_spacy}):
                adapter = loader.load(tmp_dir)
                self.assertIsInstance(adapter, SpacyModelAdapter)
                self.assertTrue(adapter.is_loaded)
                loader.unload(adapter)
                self.assertFalse(adapter.is_loaded)

    def test_generic_model_loader_dispatches_to_spacy(self):
        """Test that generic ModelLoader correctly detects and routes spaCy models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump({"name": "custom_ner_pipeline"}, f)

            env = EnvironmentManager(override_packages={"spacy": True})
            loader = ModelLoader(env_manager=env)

            mock_nlp = MagicMock()
            mock_nlp.meta = {"name": "custom_ner_pipeline"}
            mock_spacy = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            with patch.dict("sys.modules", {"spacy": mock_spacy}):
                instance = loader.load(tmp_dir)
                self.assertIsInstance(instance, SpacyModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.architecture, "spaCyPipeline")
                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_ml_load_facade_with_spacy(self):
        """Test full ml.load() facade integration for spaCy."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump({"name": "en_core_web_sm"}, f)

            env = EnvironmentManager(override_packages={"spacy": True})
            mock_nlp = MagicMock()
            mock_spacy = MagicMock()
            mock_spacy.load.return_value = mock_nlp

            with patch.dict("sys.modules", {"spacy": mock_spacy}):
                manager = ml.load(model_source=tmp_dir, framework="spacy", env_manager=env)
                self.assertTrue(manager.is_ready)
                self.assertIsInstance(manager._active_model_instance, SpacyModelAdapter)
                manager.unload()
                self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
                manager.stop()


class TestTransformersModelLoaderAndAdapter(unittest.TestCase):
    """Test suite for TransformersModelLoader and TransformersModelAdapter."""

    def test_transformers_adapter_load_and_unload_mocked(self):
        """Test TransformersModelAdapter lifecycle with mocked Transformers and PyTorch."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_hf_config = MagicMock()

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = mock_hf_config
            mock_tf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(
                    model_dir=tmp_dir,
                    device="cpu",
                    architecture="BertForSequenceClassification",
                    model_type="bert",
                )
                self.assertFalse(adapter.is_loaded)

                adapter.load()
                self.assertTrue(adapter.is_loaded)
                self.assertIs(adapter.model, mock_model)
                self.assertIs(adapter.tokenizer, mock_tokenizer)

                # Verify local_files_only=True was passed
                mock_tf.AutoConfig.from_pretrained.assert_called_with(tmp_dir, local_files_only=True)
                mock_tf.AutoModelForSequenceClassification.from_pretrained.assert_called_with(
                    tmp_dir, config=mock_hf_config, local_files_only=True
                )
                mock_tf.AutoTokenizer.from_pretrained.assert_called_with(tmp_dir, local_files_only=True)

                mock_model.to.assert_called_with("cpu")
                mock_model.eval.assert_called_once()

                res = adapter.predict("Hello World text payload")
                self.assertIsInstance(res, dict)
                self.assertEqual(res["label"], "LABEL_0")
                self.assertEqual(res["class_id"], 0)
                self.assertIn("probabilities", res)
                mock_model.assert_called()
                mock_tokenizer.assert_called()

                adapter.unload()
                self.assertFalse(adapter.is_loaded)
                self.assertIsNone(adapter.model)
                self.assertIsNone(adapter.tokenizer)

    def test_transformers_adapter_token_classification_inference_mocked(self):
        """Test token classification inference with token logits and BIO entity grouping."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            id2label = {
                "0": "O",
                "1": "B-PER",
                "2": "I-PER",
                "3": "B-ORG",
                "4": "I-ORG",
                "5": "B-LOC",
            }
            cfg = {
                "model_type": "distilbert",
                "architectures": ["DistilBertForTokenClassification"],
                "id2label": id2label,
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.architectures = ["DistilBertForTokenClassification"]
            mock_hf_config.id2label = id2label
            mock_model.config = mock_hf_config

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = mock_hf_config
            mock_tf.AutoModelForTokenClassification.from_pretrained.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            mock_tokenizer.all_special_tokens = ["[CLS]", "[SEP]"]
            mock_tokenizer.return_value = {
                "input_ids": [[101, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 102]],
                "offset_mapping": [[
                    (0, 0),    # [CLS]
                    (0, 5),    # Apple
                    (6, 11),   # hired
                    (12, 16),  # John
                    (17, 22),  # Smith
                    (23, 25),  # in
                    (26, 32),  # London
                    (32, 33),  # .
                    (0, 0),    # [SEP]
                ]],
            }
            mock_tokenizer.convert_ids_to_tokens.return_value = [
                "[CLS]", "Apple", "hired", "John", "Smith", "in", "London", ".", "[SEP]"
            ]

            mock_logits = MagicMock()
            mock_logits.tolist.return_value = [[0, 3, 0, 1, 2, 0, 5, 0, 0]]
            mock_model.return_value = MagicMock(logits=mock_logits)

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(
                    model_dir=tmp_dir,
                    device="cpu",
                    architecture="DistilBertForTokenClassification",
                    model_type="distilbert",
                )
                adapter.load()

                text = "Apple hired John Smith in London."
                res = adapter.predict(text)

                self.assertIsInstance(res, dict)
                self.assertEqual(res["text"], text)
                self.assertIn("tokens", res)
                self.assertIn("entities", res)

                tokens = res["tokens"]
                self.assertEqual(len(tokens), 9)
                self.assertEqual(tokens[1]["token"], "Apple")
                self.assertEqual(tokens[1]["label"], "B-ORG")
                self.assertEqual(tokens[1]["start"], 0)
                self.assertEqual(tokens[1]["end"], 5)

                self.assertEqual(tokens[3]["token"], "John")
                self.assertEqual(tokens[3]["label"], "B-PER")
                self.assertEqual(tokens[4]["token"], "Smith")
                self.assertEqual(tokens[4]["label"], "I-PER")

                self.assertEqual(tokens[6]["token"], "London")
                self.assertEqual(tokens[6]["label"], "B-LOC")

                entities = res["entities"]
                self.assertEqual(len(entities), 3)

                self.assertEqual(entities[0]["entity"], "ORG")
                self.assertEqual(entities[0]["word"], "Apple")
                self.assertEqual(entities[0]["start"], 0)
                self.assertEqual(entities[0]["end"], 5)

                self.assertEqual(entities[1]["entity"], "PER")
                self.assertEqual(entities[1]["word"], "John Smith")
                self.assertEqual(entities[1]["start"], 12)
                self.assertEqual(entities[1]["end"], 22)

                self.assertEqual(entities[2]["entity"], "LOC")
                self.assertEqual(entities[2]["word"], "London")
                self.assertEqual(entities[2]["start"], 26)
                self.assertEqual(entities[2]["end"], 32)

                adapter.unload()

    def test_transformers_adapter_token_classification_dict_and_batch_input_mocked(self):
        """Test token classification with dict input and batch list input."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            id2label = {"0": "O", "1": "B-ORG"}
            cfg = {"model_type": "bert", "architectures": ["BertForTokenClassification"], "id2label": id2label}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.architectures = ["BertForTokenClassification"]
            mock_hf_config.id2label = id2label
            mock_model.config = mock_hf_config

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = mock_hf_config
            mock_tf.AutoModelForTokenClassification.from_pretrained.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            mock_tokenizer.all_special_tokens = []
            mock_tokenizer.return_value = {"input_ids": [[101, 102]], "offset_mapping": [[(0, 5), (6, 11)]]}
            mock_tokenizer.convert_ids_to_tokens.return_value = ["Apple", "Corp"]
            mock_logits = MagicMock()
            mock_logits.tolist.return_value = [[1, 1]]
            mock_model.return_value = MagicMock(logits=mock_logits)

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(model_dir=tmp_dir, architecture="BertForTokenClassification")
                adapter.load()

                # Test dict input
                dict_res = adapter.predict({"text": "Apple Corp"})
                self.assertIsInstance(dict_res, dict)
                self.assertEqual(dict_res["text"], "Apple Corp")

                # Test list batch input
                batch_res = adapter.predict(["Apple Corp", "Apple Corp"])
                self.assertIsInstance(batch_res, list)
                self.assertEqual(len(batch_res), 2)
                self.assertEqual(batch_res[0]["text"], "Apple Corp")

                adapter.unload()

    def test_transformers_adapter_question_answering_inference_mocked(self):
        """Test question answering inference with question and context extraction."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertForQuestionAnswering"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.architectures = ["BertForQuestionAnswering"]

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = mock_hf_config
            mock_tf.AutoModelForQuestionAnswering.from_pretrained.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(model_dir=tmp_dir, architecture="BertForQuestionAnswering")
                adapter.load()

                payload = {"question": "Where is London?", "context": "London is the capital of the UK."}
                res = adapter.predict(payload)
                self.assertIsInstance(res, dict)
                self.assertIn("answer", res)
                self.assertIn("score", res)
                self.assertIn("start", res)
                self.assertIn("end", res)

                adapter.unload()

    def test_transformers_adapter_generic_inference_mocked(self):
        """Test generic transformers model inference returning representation metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertModel"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_hf_config = MagicMock()
            mock_hf_config.architectures = ["BertModel"]

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = mock_hf_config
            mock_tf.AutoModel.from_pretrained.return_value = mock_model
            mock_tf.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

            mock_model.return_value = MagicMock(
                last_hidden_state=MagicMock(shape=[1, 5, 768]),
                pooler_output=MagicMock(shape=[1, 768]),
            )

            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(model_dir=tmp_dir, architecture="BertModel")
                adapter.load()

                res = adapter.predict("Generic sentence payload")
                self.assertIsInstance(res, dict)
                self.assertEqual(res["last_hidden_state_shape"], [1, 5, 768])
                self.assertEqual(res["pooler_output_shape"], [1, 768])

                adapter.unload()

    def test_transformers_adapter_error_handling(self):
        """Test inference error handling across unloaded model, missing tokenizer, and invalid inputs."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "distilbert", "architectures": ["DistilBertForTokenClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            adapter = TransformersModelAdapter(
                model_dir=tmp_dir,
                architecture="DistilBertForTokenClassification",
            )

            # Unloaded model raises ModelLoadError
            with self.assertRaises(ModelLoadError):
                adapter.predict("Sample text")

            # Mock loaded without tokenizer
            adapter._is_loaded = True
            adapter.model = MagicMock()
            adapter.tokenizer = None

            with self.assertRaises(InferenceError) as cm_tok:
                adapter.predict("Sample text")
            self.assertIn("Tokenizer is required", str(cm_tok.exception))

            # Provide tokenizer
            adapter.tokenizer = MagicMock()

            # None input raises InferenceError
            with self.assertRaises(InferenceError) as cm_none:
                adapter.predict(None)
            self.assertIn("cannot be None", str(cm_none.exception))

            # Empty string input raises InferenceError
            with self.assertRaises(InferenceError) as cm_empty:
                adapter.predict("   ")
            self.assertIn("empty or whitespace", str(cm_empty.exception))

            # Empty sequence raises InferenceError
            with self.assertRaises(InferenceError) as cm_empty_seq:
                adapter.predict([])
            self.assertIn("cannot be empty", str(cm_empty_seq.exception))

            # Unsupported input type raises InferenceError
            with self.assertRaises(InferenceError) as cm_type:
                adapter.predict(12345)
            self.assertIn("Unsupported input type", str(cm_type.exception))

            # Dict missing 'text' key raises InferenceError
            with self.assertRaises(InferenceError) as cm_dict:
                adapter.predict({"wrong_key": "val"})
            self.assertIn("must contain a 'text' key", str(cm_dict.exception))

            # QA with missing context raises InferenceError
            adapter.architecture = "BertForQuestionAnswering"
            with self.assertRaises(InferenceError) as cm_qa:
                adapter.predict({"question": "Only question?"})
            self.assertIn("Context cannot be empty", str(cm_qa.exception))

    def test_transformers_adapter_task_specific_load_failure_no_generic_fallback(self):
        """Test that task-specific model load failure raises ModelLoadError and does not fallback to AutoModel."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "distilbert", "architectures": ["DistilBertForTokenClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = MagicMock(architectures=["DistilBertForTokenClassification"])
            mock_tf.AutoModelForTokenClassification.from_pretrained.side_effect = RuntimeError("Corrupted token classification weights")
            mock_tf.AutoModel.from_pretrained.return_value = MagicMock()  # Generic fallback mock

            mock_torch = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = TransformersModelAdapter(
                    model_dir=tmp_dir,
                    architecture="DistilBertForTokenClassification",
                )
                with self.assertRaises(ModelLoadError) as cm:
                    adapter.load()
                self.assertIn("Corrupted token classification weights", str(cm.exception))
                # Ensure AutoModel fallback was NOT called for task-specific architecture
                mock_tf.AutoModel.from_pretrained.assert_not_called()

    def test_transformers_adapter_missing_dependency(self):
        """Test TransformersModelAdapter raises DependencyMissingError when torch/transformers missing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter = TransformersModelAdapter(model_dir=tmp_dir)
            with patch.dict("sys.modules", {"torch": None, "transformers": None}):
                with self.assertRaises(DependencyMissingError) as cm:
                    adapter.load()
                self.assertIn("torch", cm.exception.missing_packages)
                self.assertIn("pip install", str(cm.exception))

    def test_transformers_loader_standalone(self):
        """Test TransformersModelLoader directly as a standalone specialized loader."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "roberta", "architectures": ["RobertaForSequenceClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={"torch": True, "transformers": True})
            loader = TransformersModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = MagicMock()
            mock_tf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
            mock_torch = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                adapter = loader.load(tmp_dir)
                self.assertIsInstance(adapter, TransformersModelAdapter)
                self.assertTrue(adapter.is_loaded)
                loader.unload(adapter)
                self.assertFalse(adapter.is_loaded)

    def test_generic_model_loader_dispatches_to_transformers(self):
        """Test that generic ModelLoader correctly detects and routes Transformers models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={"torch": True, "transformers": True})
            loader = ModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = MagicMock()
            mock_tf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
            mock_torch = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                instance = loader.load(tmp_dir)
                self.assertIsInstance(instance, TransformersModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.architecture, "BertForSequenceClassification")
                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_generic_model_loader_dispatches_to_transformers_with_config_cfg_present(self):
        """Test that generic ModelLoader routes to TransformersModelAdapter when config.cfg is present in the directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "distilbert", "architectures": ["DistilBertForTokenClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w") as f:
                f.write("[unrelated]\nkey=value\n")
            with open(os.path.join(tmp_dir, "model.safetensors"), "wb") as f:
                f.write(b"dummy")

            env = EnvironmentManager(override_packages={"torch": True, "transformers": True, "spacy": True})
            loader = ModelLoader(env_manager=env)

            mock_model = MagicMock()
            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = MagicMock()
            mock_tf.AutoModelForTokenClassification.from_pretrained.return_value = mock_model
            mock_torch = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                instance = loader.load(tmp_dir)
                self.assertIsInstance(instance, TransformersModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.architecture, "DistilBertForTokenClassification")
                self.assertEqual(instance.model_type, "distilbert")
                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_ml_load_facade_with_transformers(self):
        """Test full ml.load() facade integration for Transformers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertForSequenceClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager(override_packages={"torch": True, "transformers": True})
            mock_model = MagicMock()
            mock_tf = MagicMock()
            mock_tf.AutoConfig.from_pretrained.return_value = MagicMock()
            mock_tf.AutoModelForSequenceClassification.from_pretrained.return_value = mock_model
            mock_torch = MagicMock()

            with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_tf}):
                manager = ml.load(model_source=tmp_dir, framework="transformers", env_manager=env)
                self.assertTrue(manager.is_ready)
                self.assertIsInstance(manager._active_model_instance, TransformersModelAdapter)
                manager.unload()
                self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
                manager.stop()


class TestONNXModelLoaderAndAdapter(unittest.TestCase):
    """Test suite for ONNXModelLoader and ONNXModelAdapter."""

    def test_onnx_adapter_load_from_single_file_mocked(self):
        """Test ONNXModelAdapter lifecycle with standalone .onnx file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_ort.InferenceSession.return_value = mock_session

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                adapter = ONNXModelAdapter(model_dir=tmp_file.name, device="cpu")
                self.assertFalse(adapter.is_loaded)

                adapter.load()
                self.assertTrue(adapter.is_loaded)
                self.assertIs(adapter.session, mock_session)

                res = adapter.predict({"input_ids": [101, 2054, 102]})
                self.assertEqual(res, {"input_ids": [101, 2054, 102]})

                adapter.unload()
                self.assertFalse(adapter.is_loaded)
                self.assertIsNone(adapter.session)

    def test_onnx_adapter_load_from_directory_mocked(self):
        """Test ONNXModelAdapter loading model.onnx from directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = os.path.join(tmp_dir, "model.onnx")
            with open(onnx_path, "wb") as f:
                f.write(b"dummy_onnx_bytes")

            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_ort.InferenceSession.return_value = mock_session

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                adapter = ONNXModelAdapter(model_dir=tmp_dir)
                adapter.load()
                self.assertTrue(adapter.is_loaded)
                self.assertEqual(adapter.onnx_file, onnx_path)
                adapter.unload()

    def test_onnx_adapter_missing_dependency(self):
        """Test ONNXModelAdapter raises DependencyMissingError when onnxruntime is missing."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            adapter = ONNXModelAdapter(model_dir=tmp_file.name)
            with patch.dict("sys.modules", {"onnxruntime": None}):
                with self.assertRaises(DependencyMissingError) as cm:
                    adapter.load()
                self.assertIn("onnxruntime", cm.exception.missing_packages)
                self.assertIn("pip install", str(cm.exception))

    def test_onnx_adapter_cuda_provider_missing(self):
        """Test requesting CUDA when CUDAExecutionProvider is absent raises HardwareIncompatibleError."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                adapter = ONNXModelAdapter(model_dir=tmp_file.name, device="cuda")
                with self.assertRaises(HardwareIncompatibleError) as cm:
                    adapter.load()
                self.assertIn("CUDAExecutionProvider", str(cm.exception))

    def test_onnx_loader_standalone(self):
        """Test ONNXModelLoader directly as a standalone specialized loader."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            env = EnvironmentManager(override_packages={"onnxruntime": True})
            loader = ONNXModelLoader(env_manager=env)

            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_ort.InferenceSession.return_value = mock_session

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                adapter = loader.load(tmp_file.name)
                self.assertIsInstance(adapter, ONNXModelAdapter)
                self.assertTrue(adapter.is_loaded)
                loader.unload(adapter)
                self.assertFalse(adapter.is_loaded)

    def test_generic_model_loader_dispatches_to_onnx(self):
        """Test that generic ModelLoader correctly detects and routes standalone ONNX file."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            env = EnvironmentManager(override_packages={"onnxruntime": True})
            loader = ModelLoader(env_manager=env)

            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_ort.InferenceSession.return_value = mock_session

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                instance = loader.load(tmp_file.name)
                self.assertIsInstance(instance, ONNXModelAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.architecture, "ONNXModel")
                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_ml_load_facade_with_onnx(self):
        """Test full ml.load() facade integration for ONNX."""
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_file:
            env = EnvironmentManager(override_packages={"onnxruntime": True})
            mock_session = MagicMock()
            mock_ort = MagicMock()
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            mock_ort.InferenceSession.return_value = mock_session

            with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
                manager = ml.load(model_source=tmp_file.name, framework="onnx", env_manager=env)
                self.assertTrue(manager.is_ready)
                self.assertIsInstance(manager._active_model_instance, ONNXModelAdapter)
                manager.unload()
                self.assertEqual(manager.state, ExecutionState.UNINITIALIZED)
                manager.stop()


class TestGenericModelLoaderDispatchAndExtensibility(unittest.TestCase):
    """Test suite for ModelLoader dispatch, extensibility, unsupported formats, and cleanup."""

    def test_empty_model_source_raises_load_error(self):
        """Test loading with empty model_source raises ModelLoadError."""
        loader = ModelLoader()
        with self.assertRaises(ModelLoadError) as cm:
            loader.load("")
        self.assertIn("Model source cannot be empty", str(cm.exception))

    def test_missing_local_model_path_raises_load_error(self):
        """Test loading non-existent path raises ModelLoadError."""
        loader = ModelLoader()
        with self.assertRaises(ModelLoadError) as cm:
            loader.load("/path/to/missing_model_artifact_xyz")
        self.assertIn("Local model path does not exist", str(cm.exception))

    def test_unsupported_framework_raises_model_format_error(self):
        """Test that an unknown/unsupported framework raises ModelFormatError listing supported frameworks."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump({"model_type": "unknown_framework_model", "architectures": ["UnknownModel"]}, f)

            env = EnvironmentManager(override_packages={"torch": True, "transformers": True})
            loader = ModelLoader(env_manager=env)
            with patch.object(
                loader.format_detector, "detect", return_value=("unsupported_framework_xyz", "UnknownModel", "unknown")
            ):
                with self.assertRaises(ModelFormatError) as cm:
                    loader.load(tmp_dir)
                self.assertIn("Model framework 'unsupported_framework_xyz' is not supported", str(cm.exception))
                self.assertIn("Currently supported frameworks", str(cm.exception))

    def test_get_supported_frameworks(self):
        """Test get_supported_frameworks returns all standard framework identifiers."""
        loader = ModelLoader()
        supported = loader.get_supported_frameworks()
        self.assertIn("speech", supported)
        self.assertIn("spacy", supported)
        self.assertIn("transformers", supported)
        self.assertIn("onnx", supported)

    def test_custom_loader_registration_and_dispatch(self):
        """Test registering a custom specialized loader dynamically extends ModelLoader."""
        class CustomFrameworkAdapter(BaseModelAdapter):
            def __init__(self, model_dir, **kwargs):
                self.model_dir = model_dir
                self._is_loaded = False
            @property
            def is_loaded(self):
                return self._is_loaded
            def load(self):
                self._is_loaded = True
                return self
            def unload(self):
                self._is_loaded = False
            def predict(self, x):
                return f"custom_pred:{x}"

        class CustomFrameworkLoader(BaseModelLoader):
            def __init__(self, **kwargs):
                pass
            def load(self, model_source, config=None):
                adapter = CustomFrameworkAdapter(model_dir=model_source)
                return adapter.load()
            def unload(self, model_instance):
                model_instance.unload()

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "custom.meta"), "w") as f:
                f.write("custom_weights")

            loader = ModelLoader()
            loader.register_loader("custom_fw", CustomFrameworkLoader)
            self.assertIn("custom_fw", loader.get_supported_frameworks())

            with patch.object(
                loader.format_detector, "detect", return_value=("custom_fw", "CustomArch", "custom_fw")
            ):
                instance = loader.load(tmp_dir)
                self.assertIsInstance(instance, CustomFrameworkAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.predict("hello"), "custom_pred:hello")

                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_custom_adapter_registration_and_dispatch(self):
        """Test registering a custom adapter class dynamically extends ModelLoader."""
        class CustomAdapter(BaseModelAdapter):
            def __init__(self, model_dir, device="cpu", **kwargs):
                self.model_dir = model_dir
                self.device = device
                self._is_loaded = False
            @property
            def is_loaded(self):
                return self._is_loaded
            def load(self):
                self._is_loaded = True
                return self
            def unload(self):
                self._is_loaded = False
            def predict(self, x):
                return f"adapter_pred:{x}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "custom.meta"), "w") as f:
                f.write("custom")

            loader = ModelLoader()
            loader.register_adapter("custom_adapter_fw", CustomAdapter)
            self.assertIn("custom_adapter_fw", loader.get_supported_frameworks())

            with patch.object(
                loader.format_detector, "detect", return_value=("custom_adapter_fw", "CustomArch", "custom_fw")
            ):
                instance = loader.load(tmp_dir)
                self.assertIsInstance(instance, CustomAdapter)
                self.assertTrue(instance.is_loaded)
                self.assertEqual(instance.predict("foo"), "adapter_pred:foo")

                loader.unload(instance)
                self.assertFalse(instance.is_loaded)

    def test_remote_artifact_temp_dir_cleaned_on_download_failure(self):
        """Verify that temporary download directories are cleaned up if S3 download fails."""
        loader = ModelLoader()
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.download_file.side_effect = RuntimeError("S3 Network Drop")
        mock_boto3.client.return_value = mock_client

        with patch.dict("sys.modules", {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.config": MagicMock()}):
            with self.assertRaises(Exception):
                loader.load("s3://test-bucket/model.tar.gz")

        self.assertEqual(loader._temp_dirs, [])

    def test_remote_artifact_temp_dir_cleaned_on_load_validation_failure(self):
        """Verify that downloaded temp directory is removed if downstream validation fails."""
        loader = ModelLoader()
        created_temp_dirs = []

        def mock_download(storage_uri):
            d = tempfile.mkdtemp(prefix="ml_framework_model_test_")
            created_temp_dirs.append(d)
            loader._temp_dirs.append(d)
            return d

        with patch.object(loader, "_download_remote_artifact", side_effect=mock_download):
            with self.assertRaises(ModelFormatError):
                loader.load("s3://test-bucket/corrupt_model.tar.gz")

        self.assertEqual(loader._temp_dirs, [])
        for d in created_temp_dirs:
            self.assertFalse(os.path.exists(d), f"Expected temp dir {d} to be deleted on failure")


if __name__ == "__main__":
    unittest.main()
