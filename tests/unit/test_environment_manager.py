"""
Unit tests for Environment Manager: dependency inspection, offline environment validation,
and hardware accelerator checks.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig, StorageConfig
from ml_framework.environment import EnvironmentManager
from ml_framework.errors import (
    DependencyMissingError,
    HardwareIncompatibleError,
)


class TestEnvironmentManager(unittest.TestCase):
    """Test suite for EnvironmentManager offline validation."""

    def test_inspect_spacy_directory(self):
        """Test inspecting a spaCy model directory with meta.json."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = {"spacy_version": "3.7.0", "pipeline": ["ner"]}
            with open(os.path.join(tmp_dir, "meta.json"), "w") as f:
                json.dump(meta, f)

            env = EnvironmentManager()
            deps = env.inspect_dependencies(tmp_dir)
            self.assertEqual(deps, ["spacy"])

    def test_inspect_transformers_directory(self):
        """Test inspecting a HuggingFace directory with config.json without .onnx."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "bert", "architectures": ["BertForTokenClassification"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager()
            deps = env.inspect_dependencies(tmp_dir)
            self.assertEqual(sorted(deps), ["torch", "transformers"])

    def test_inspect_onnx_directory(self):
        """Test inspecting an ONNX model directory with config.json and .onnx file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump({"model_type": "bert"}, f)
            with open(os.path.join(tmp_dir, "model.onnx"), "w") as f:
                f.write("dummy onnx")

            env = EnvironmentManager()
            deps = env.inspect_dependencies(tmp_dir)
            self.assertEqual(deps, ["onnxruntime"])

    def test_inspect_storage_source(self):
        """Test that remote object storage source triggers boto3 dependency."""
        env = EnvironmentManager()
        deps = env.inspect_dependencies("seaweedfs://models/ner.tar.gz")
        self.assertIn("boto3", deps)

    def test_missing_dependency_reporting_with_offline_install_command(self):
        """
        Verify that missing dependencies raise DependencyMissingError
        with clear offline installation instructions.
        """
        env = EnvironmentManager(override_packages={"spacy": False})
        cfg = FrameworkConfig(framework="spacy")

        with self.assertRaises(DependencyMissingError) as cm:
            env.validate_environment(cfg)

        err = cm.exception
        self.assertIn("spacy", err.missing_packages)
        self.assertIn("Offline installation recommendation", str(err))
        self.assertIn("pip install --no-index --find-links=/path/to/offline/wheels", str(err))

    def test_successful_environment_validation(self):
        """Test validation succeeds when dependencies are satisfied."""
        env = EnvironmentManager(override_packages={"spacy": True})
        cfg = FrameworkConfig(framework="spacy", device="cpu")
        self.assertTrue(env.validate_environment(cfg))

    def test_hardware_validation_cpu(self):
        """Test CPU device is always validated."""
        env = EnvironmentManager()
        env.validate_hardware("cpu")
        env.validate_hardware("auto")

    def test_hardware_validation_cuda_missing_torch(self):
        """Test requesting CUDA when PyTorch is not installed."""
        env = EnvironmentManager(override_packages={"torch": False})
        with self.assertRaises(HardwareIncompatibleError) as cm:
            env.validate_hardware("cuda")
        self.assertIn("PyTorch ('torch') is not installed", str(cm.exception))

    def test_hardware_validation_cuda_unavailable(self):
        """Test requesting CUDA when no CUDA GPU is present."""
        env = EnvironmentManager(override_packages={"torch": True})
        with patch("importlib.import_module") as mock_import:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with self.assertRaises(HardwareIncompatibleError) as cm:
                    env.validate_hardware("cuda")
                self.assertIn("no CUDA-compatible GPU is available", str(cm.exception))

    def test_unsupported_device_type(self):
        """Test invalid device parameter."""
        env = EnvironmentManager()
        with self.assertRaises(HardwareIncompatibleError):
            env.validate_hardware("tpu_v5")

    def test_inspect_standalone_onnx_file(self):
        """Test that a standalone .onnx file path triggers onnxruntime dependency."""
        env = EnvironmentManager()
        deps = env.inspect_dependencies("/opt/models/sentiment_classifier.onnx")
        self.assertEqual(deps, ["onnxruntime"])

    def test_inspect_speech_directory_whisper(self):
        """Test that a model directory with whisper model_type triggers speech dependencies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {
                "model_type": "whisper",
                "architectures": ["WhisperForConditionalGeneration"]
            }
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)

            env = EnvironmentManager()
            deps = env.inspect_dependencies(tmp_dir)
            self.assertEqual(sorted(deps), ["numpy", "soundfile", "torch", "transformers"])

    def test_inspect_speech_directory_preprocessor(self):
        """Test that a directory with preprocessor_config.json triggers speech dependencies."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg = {"model_type": "custom_asr", "architectures": ["CustomASRModel"]}
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(cfg, f)
            pre = {"feature_extractor_type": "Wav2Vec2FeatureExtractor"}
            with open(os.path.join(tmp_dir, "preprocessor_config.json"), "w") as f:
                json.dump(pre, f)

            env = EnvironmentManager()
            deps = env.inspect_dependencies(tmp_dir)
            self.assertIn("soundfile", deps)
            self.assertIn("numpy", deps)

    def test_inspect_metadata_dictionary(self):
        """Test inspecting dependency requirements from a metadata dictionary."""
        env = EnvironmentManager()
        meta = {
            "framework": "speech",
            "format": "onnx",
            "requires_storage": True,
            "requires_kafka": True,
            "dependencies": ["custom_lib"],
        }
        deps = env.inspect_dependencies(meta)
        self.assertIn("soundfile", deps)
        self.assertIn("onnxruntime", deps)
        self.assertIn("boto3", deps)
        self.assertIn("confluent_kafka", deps)
        self.assertIn("custom_lib", deps)

    def test_hardware_validation_cuda_device_index_valid(self):
        """Test CUDA device index (e.g. cuda:0) when CUDA is available."""
        env = EnvironmentManager(override_packages={"torch": True})
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        with patch.dict("sys.modules", {"torch": mock_torch}):
            env.validate_hardware("cuda:0")
            env.validate_hardware("cuda:1")

    def test_hardware_validation_cuda_device_index_out_of_bounds(self):
        """Test CUDA device index exceeds available device count."""
        env = EnvironmentManager(override_packages={"torch": True})
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with self.assertRaises(HardwareIncompatibleError) as cm:
                env.validate_hardware("cuda:2")
            self.assertIn("CUDA device index 2 requested, but only 1 CUDA device(s) are available", str(cm.exception))

    def test_hardware_validation_cuda_device_index_invalid_format(self):
        """Test CUDA device index with non-integer format."""
        env = EnvironmentManager(override_packages={"torch": True})
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with self.assertRaises(HardwareIncompatibleError) as cm:
                env.validate_hardware("cuda:first_gpu")
            self.assertIn("Invalid CUDA device index", str(cm.exception))

    def test_hardware_validation_mps_unavailable(self):
        """Test Apple Silicon MPS acceleration when unavailable."""
        env = EnvironmentManager(override_packages={"torch": True})
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            with self.assertRaises(HardwareIncompatibleError) as cm:
                env.validate_hardware("mps")
            self.assertIn("MPS is not available", str(cm.exception))

    def test_device_resolution(self):
        """Test resolve_device and get_device behavior."""
        env = EnvironmentManager()
        # Explicit CPU
        self.assertEqual(env.resolve_device("cpu"), "cpu")
        self.assertEqual(env.get_device("cpu"), "cpu")

        # Auto resolution
        auto_dev = env.resolve_device("auto")
        self.assertIn(auto_dev, {"cpu", "cuda", "mps"})

        # Configuration resolution in validate_environment
        cfg = FrameworkConfig(framework="spacy", device="cpu")
        env_with_override = EnvironmentManager(override_packages={"spacy": True})
        env_with_override.validate_environment(cfg)
        self.assertEqual(env_with_override.get_device(), "cpu")

    def test_validate_environment_with_s3_input_source(self):
        """Test that s3:// input source triggers boto3 dependency check during environment validation."""
        env = EnvironmentManager(override_packages={"boto3": False})
        cfg = FrameworkConfig(
            input_source="s3://raw-bucket/data.jsonl",
            device="cpu"
        )
        with self.assertRaises(DependencyMissingError) as cm:
            env.validate_environment(cfg)
        self.assertIn("boto3", cm.exception.missing_packages)

    def test_validate_environment_with_kafka_input_source(self):
        """Test that kafka:// input source triggers confluent_kafka dependency check."""
        env = EnvironmentManager(override_packages={"confluent_kafka": False})
        cfg = FrameworkConfig(
            input_source="kafka://stream-topic",
            device="cpu"
        )
        with self.assertRaises(DependencyMissingError) as cm:
            env.validate_environment(cfg)
        self.assertIn("confluent_kafka", cm.exception.missing_packages)

    def test_package_to_module_mapping(self):
        """Test package name normalization using find_spec."""
        env = EnvironmentManager()
        # Verify package normalization handling
        self.assertIsInstance(env.is_package_installed("confluent-kafka"), bool)
        self.assertIsInstance(env.is_package_installed("torch"), bool)

    def test_override_device_simulation(self):
        """Test simulated hardware environment override."""
        env = EnvironmentManager(override_device="cuda")
        self.assertEqual(env.resolve_device("auto"), "cuda")
        self.assertEqual(env.get_device(), "cuda")
        env.validate_hardware("cuda")

    def test_close_lifecycle(self):
        """Test close() lifecycle method executes without error."""
        env = EnvironmentManager()
        env.close()

    def test_inspect_transformers_directory_with_config_cfg_does_not_require_spacy(self):
        """
        Verify that a Transformers model directory containing config.json, config.cfg,
        and model.safetensors only requires ['torch', 'transformers'] and not 'spacy'.
        """
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
                f.write(b"dummy_weights")

            env = EnvironmentManager()

            # Automatic framework inspection
            deps_auto = env.inspect_dependencies(tmp_dir)
            self.assertEqual(sorted(deps_auto), ["torch", "transformers"])
            self.assertNotIn("spacy", deps_auto)

            # Explicit framework=transformers
            deps_tf = env.inspect_dependencies(tmp_dir, framework="transformers")
            self.assertEqual(sorted(deps_tf), ["torch", "transformers"])
            self.assertNotIn("spacy", deps_tf)

    def test_inspect_genuine_spacy_directory_with_meta_and_config_cfg(self):
        """
        Verify that a genuine spaCy directory containing meta.json and config.cfg
        requires ['spacy'] and not transformers dependencies.
        """
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

            env = EnvironmentManager()

            # Automatic framework inspection
            deps_auto = env.inspect_dependencies(tmp_dir)
            self.assertEqual(deps_auto, ["spacy"])

            # Explicit framework=spacy
            deps_spacy = env.inspect_dependencies(tmp_dir, framework="spacy")
            self.assertEqual(deps_spacy, ["spacy"])

    def test_validate_environment_transformers_with_config_cfg_without_spacy(self):
        """
        Verify that environment validation passes for a Transformers model containing
        an incidental config.cfg even when spaCy is not installed in the environment.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            cfg_dict = {
                "model_type": "distilbert",
                "architectures": ["DistilBertForTokenClassification"],
            }
            with open(os.path.join(tmp_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg_dict, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[unrelated]\nkey=value\n")
            with open(os.path.join(tmp_dir, "model.safetensors"), "wb") as f:
                f.write(b"dummy")

            # Environment where torch and transformers are present, but spacy is NOT installed
            env = EnvironmentManager(override_packages={
                "torch": True,
                "transformers": True,
                "spacy": False,
            })
            framework_cfg = FrameworkConfig(model_source=tmp_dir, device="cpu")
            self.assertTrue(env.validate_environment(framework_cfg))

    def test_validate_environment_spacy_model_missing_spacy_raises_error(self):
        """
        Verify that environment validation fails with DependencyMissingError for a genuine
        spaCy model when spaCy is not installed.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            meta = {"name": "en_core_web_sm", "spacy_version": ">=3.7.0"}
            with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f)
            with open(os.path.join(tmp_dir, "config.cfg"), "w", encoding="utf-8") as f:
                f.write("[nlp]\nlang = \"en\"\n")

            env = EnvironmentManager(override_packages={"spacy": False})
            framework_cfg = FrameworkConfig(model_source=tmp_dir, device="cpu")
            with self.assertRaises(DependencyMissingError) as cm:
                env.validate_environment(framework_cfg)
            self.assertIn("spacy", cm.exception.missing_packages)


if __name__ == "__main__":
    unittest.main()
