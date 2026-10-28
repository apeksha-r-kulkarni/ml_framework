"""
Unit Tests for Configuration-Driven Model Selection Across Model Families.

Covers all model families:
1. NeMo Conformer-CTC (stt_en_conformer_ctc_large.nemo)
2. NeMo Language Detection (AmberNet & PearlNet)
3. Wav2Vec2 ASR (wav2vec2_large_960h)
4. MMS Language Detection (model/Language_Detection/1)
5. Standalone .safetensors rejection
6. SpeechBrain VoxLingua107 (model/Language_Detection/4)
7. Faster Whisper & Whisper variants (model/Transcription/faster_whisper_*, whisper_*)
8. DeepFilterNet3 (model/Denoiser/DeepFilterNet3)
9. Task contracts (transcription, language_detection, denoising)
10. Model switching via configuration without code modifications
"""

import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from ml_framework import ml, ExecutionState, FrameworkConfig, StorageConfig, TritonConfig
from ml_framework.models.format_detector import FormatDetector
from ml_framework.models.model_loader import ModelLoader
from ml_framework.inference.tensor_mapper import TensorMapper
from ml_framework.inference.triton_engine import TritonInferenceEngine
from ml_framework.errors.exceptions import (
    ConfigurationError,
    ModelFormatError,
    ModelLoadError,
    InputProcessingError,
    DependencyMissingError,
    InferenceError,
)


class TestModelFamiliesConfig(unittest.TestCase):
    """Unit tests for configuration-driven selection and validation across model families."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="ml_framework_test_families_")
        self.detector = FormatDetector()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Configuration Parsing and Validation
    # -------------------------------------------------------------------------

    def test_config_driven_selection_fields(self):
        """Verify configuration parses bucket, object key, model_dir, task, and input_type."""
        cfg = FrameworkConfig.from_dict({
            "model_bucket": "speech-data",
            "model_key": "model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo",
            "model_id": "nemo_conformer_python",
            "task": "transcription",
            "input_type": "audio",
        })
        self.assertEqual(cfg.model_bucket, "speech-data")
        self.assertEqual(cfg.model_key, "model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo")
        self.assertEqual(cfg.model_name, "nemo_conformer_python")
        self.assertEqual(cfg.task, "transcription")
        self.assertEqual(cfg.input_type, "audio")
        self.assertEqual(
            cfg.model_source,
            "s3://speech-data/model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo"
        )

    def test_config_aliases_resolution(self):
        """Verify configuration aliases: bucket, object_key, model_dir, MODEL_TASK, etc."""
        cfg = FrameworkConfig.from_dict({
            "MINIO_BUCKET": "speech-data",
            "model_dir": "model/Denoiser/DeepFilterNet3/",
            "MODEL_ID": "deepfilter3",
            "TASK": "denoising",
            "INPUT_TYPE": "audio",
        })
        self.assertEqual(cfg.model_bucket, "speech-data")
        self.assertEqual(cfg.model_key, "model/Denoiser/DeepFilterNet3/")
        self.assertEqual(cfg.model_name, "deepfilter3")
        self.assertEqual(cfg.task, "denoising")
        self.assertEqual(cfg.input_type, "audio")

    def test_config_validation_invalid_task(self):
        """Verify ConfigurationError on invalid task specification."""
        with self.assertRaises(ConfigurationError) as ctx:
            FrameworkConfig.from_dict({"task": "image_classification"})
        self.assertIn("Invalid task", str(ctx.exception))

    def test_config_validation_conflicting_task_and_input(self):
        """Verify ConfigurationError when task and input_type conflict."""
        with self.assertRaises(ConfigurationError) as ctx:
            FrameworkConfig.from_dict({"task": "denoising", "input_type": "text"})
        self.assertIn("denoising", str(ctx.exception))
        self.assertIn("text", str(ctx.exception))

        with self.assertRaises(ConfigurationError) as ctx2:
            FrameworkConfig.from_dict({"task": "transcription", "input_type": "text"})
        self.assertIn("transcription", str(ctx2.exception))

    # -------------------------------------------------------------------------
    # 2. NeMo Conformer-CTC (Transcription)
    # -------------------------------------------------------------------------

    def test_nemo_conformer_format_and_task(self):
        """Verify NeMo Conformer-CTC detection, task mapping, and backend compatibility."""
        nemo_file = os.path.join(self.temp_dir, "stt_en_conformer_ctc_large.nemo")
        with open(nemo_file, "wb") as f:
            f.write(b"PK\x03\x04fake_nemo_archive")

        fw, arch, mtype = self.detector.detect(nemo_file)
        self.assertEqual(fw, "nemo")
        self.assertEqual(arch, "EncDecCTCModelBPE")
        self.assertEqual(mtype, "stt_en_conformer_ctc_large")

        task = self.detector.detect_task(nemo_file, architecture=arch, model_type=mtype)
        self.assertEqual(task, "transcription")

        loader = ModelLoader()
        # NeMo Conformer is compatible with python / nemo_conformer_python backend
        fw_out, arch_out, mtype_out = loader.validate_backend_compatibility(
            nemo_file, backend="nemo_conformer_python", model_name="nemo_conformer_python"
        )
        self.assertEqual(fw_out, "nemo")
        self.assertEqual(arch_out, "EncDecCTCModelBPE")

    def test_nemo_conformer_tensor_contracts(self):
        """Verify transcription tensor contract: audio in -> TRANSCRIPT text out."""
        mapper = TensorMapper(task="transcription")
        dummy_audio = np.zeros(16000, dtype=np.float32)

        inputs, outputs = mapper.build_request(dummy_audio, model_name="nemo_conformer_python")
        self.assertEqual(len(inputs), 2)
        self.assertEqual(inputs[0]["name"], "AUDIO_SIGNAL")
        self.assertEqual(inputs[0]["datatype"], "FP32")
        self.assertEqual(inputs[1]["name"], "SAMPLE_RATE")
        self.assertEqual(inputs[1]["data"], [16000])
        self.assertEqual(outputs, [{"name": "TRANSCRIPT"}])

        mock_triton_response = {
            "model_name": "nemo_conformer_python",
            "outputs": [
                {"name": "TRANSCRIPT", "datatype": "BYTES", "shape": [1], "data": ["welcome to speech recognition"]}
            ]
        }
        res = mapper.parse_response(mock_triton_response, task="transcription")
        self.assertEqual(res["task"], "transcription")
        self.assertEqual(res["text"], "welcome to speech recognition")
        self.assertEqual(res["transcription"], "welcome to speech recognition")

    # -------------------------------------------------------------------------
    # 3. NeMo Language Detection (AmberNet & PearlNet)
    # -------------------------------------------------------------------------

    def test_nemo_lid_ambernet_pearlnet_format_and_backend_rejection(self):
        """Verify AmberNet and PearlNet detection as EncDecSpeakerLabelModel and rejection on ASR CTC backend."""
        amber_file = os.path.join(self.temp_dir, "ambernet.nemo")
        with open(amber_file, "wb") as f:
            f.write(b"PK\x03\x04ambernet_archive")

        pearl_file = os.path.join(self.temp_dir, "pearlnet.nemo")
        with open(pearl_file, "wb") as f:
            f.write(b"PK\x03\x04pearlnet_archive")

        # Format detection
        fw_a, arch_a, mtype_a = self.detector.detect(amber_file)
        self.assertEqual(fw_a, "nemo")
        self.assertEqual(arch_a, "EncDecSpeakerLabelModel")
        self.assertEqual(mtype_a, "nemo_lid")

        fw_p, arch_p, mtype_p = self.detector.detect(pearl_file)
        self.assertEqual(fw_p, "nemo")
        self.assertEqual(arch_p, "EncDecSpeakerLabelModel")

        # Task detection
        self.assertEqual(self.detector.detect_task(amber_file, architecture=arch_a, model_type=mtype_a), "language_detection")
        self.assertEqual(self.detector.detect_task(pearl_file, architecture=arch_p, model_type=mtype_p), "language_detection")

        loader = ModelLoader()
        # Must be rejected on nemo_conformer_python / ASR backend
        with self.assertRaises(ModelFormatError) as ctx:
            loader.validate_backend_compatibility(amber_file, backend="nemo_conformer_python", model_name="nemo_conformer_python")
        self.assertIn("EncDecSpeakerLabelModel", str(ctx.exception))
        self.assertIn("incompatible", str(ctx.exception).lower())

        # Acceptance on dedicated LID backend
        fw_ok, arch_ok, _ = loader.validate_backend_compatibility(amber_file, backend="nemo_lid_python", model_name="ambernet")
        self.assertEqual(fw_ok, "nemo")
        self.assertEqual(arch_ok, "EncDecSpeakerLabelModel")

    def test_nemo_lid_tensor_contracts(self):
        """Verify language detection tensor contract: audio in -> language label out."""
        mapper = TensorMapper(task="language_detection")
        dummy_audio = np.zeros(8000, dtype=np.float32)

        inputs, outputs = mapper.build_request(dummy_audio, model_name="ambernet")
        self.assertEqual(inputs[0]["name"], "AUDIO_SIGNAL")
        self.assertEqual(outputs, [{"name": "LANGUAGE"}])

        mock_triton_response = {
            "model_name": "ambernet",
            "outputs": [
                {"name": "LANGUAGE", "datatype": "BYTES", "shape": [1], "data": ["en"]},
                {"name": "CONFIDENCE", "datatype": "FP32", "shape": [1], "data": [0.97]},
            ]
        }
        res = mapper.parse_response(mock_triton_response, task="language_detection")
        self.assertEqual(res["task"], "language_detection")
        self.assertEqual(res["language"], "en")
        self.assertEqual(res["label"], "en")
        self.assertAlmostEqual(res["confidence"], 0.97, places=2)

    # -------------------------------------------------------------------------
    # 4. Wav2Vec2 ASR (wav2vec2_large_960h)
    # -------------------------------------------------------------------------

    def test_wav2vec2_asr_format_and_backend_validation(self):
        """Verify Wav2Vec2 CTC detection and rejection on NeMo Python backend."""
        model_dir = os.path.join(self.temp_dir, "wav2vec2_large_960h")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForCTC"]}, f)
        with open(os.path.join(model_dir, "pytorch_model.bin"), "wb") as f:
            f.write(b"fake_weights")
        with open(os.path.join(model_dir, "vocab.json"), "w") as f:
            json.dump({"<pad>": 0, "a": 1}, f)

        fw, arch, mtype = self.detector.detect(model_dir)
        self.assertEqual(fw, "speech")
        self.assertEqual(arch, "Wav2Vec2ForCTC")
        self.assertEqual(mtype, "wav2vec2")
        self.assertEqual(self.detector.detect_task(model_dir, architecture=arch, model_type=mtype), "transcription")

        loader = ModelLoader()
        # NeMo Conformer ASR backend must reject Transformers artifact
        with self.assertRaises(ModelFormatError) as ctx:
            loader.validate_backend_compatibility(model_dir, backend="nemo_conformer_python", model_name="nemo_conformer_python")
        self.assertIn("incompatible", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 5. MMS Language Detection (model/Language_Detection/1)
    # -------------------------------------------------------------------------

    def test_mms_language_detection_format(self):
        """Verify MMS Wav2Vec2 SequenceClassification detection and task mapping."""
        mms_dir = os.path.join(self.temp_dir, "mms_lid")
        os.makedirs(mms_dir, exist_ok=True)
        with open(os.path.join(mms_dir, "config.json"), "w") as f:
            json.dump({"model_type": "wav2vec2", "architectures": ["Wav2Vec2ForSequenceClassification"]}, f)
        with open(os.path.join(mms_dir, "langs.txt"), "w") as f:
            f.write("eng\nfra\ndeu\nspa\n")
        with open(os.path.join(mms_dir, "model.safetensors"), "wb") as f:
            f.write(b"fake_safetensors_data")

        fw, arch, mtype = self.detector.detect(mms_dir)
        self.assertEqual(fw, "transformers")
        self.assertEqual(arch, "Wav2Vec2ForSequenceClassification")
        self.assertEqual(mtype, "wav2vec2_seq_class")

        task = self.detector.detect_task(mms_dir, architecture=arch, model_type=mtype)
        self.assertEqual(task, "language_detection")

    # -------------------------------------------------------------------------
    # 6. Standalone .safetensors Rejection
    # -------------------------------------------------------------------------

    def test_standalone_safetensors_file_rejection(self):
        """Verify standalone .safetensors file is rejected without config.json metadata."""
        st_file = os.path.join(self.temp_dir, "model.safetensors")
        with open(st_file, "wb") as f:
            f.write(b"standalone_weights")

        with self.assertRaises(ModelFormatError) as ctx:
            self.detector.detect(st_file)
        self.assertIn("Standalone .safetensors files cannot be loaded", str(ctx.exception))
        self.assertIn("config.json", str(ctx.exception))

    def test_directory_only_safetensors_rejection(self):
        """Verify directory containing only .safetensors without config.json is rejected."""
        st_dir = os.path.join(self.temp_dir, "st_only_dir")
        os.makedirs(st_dir, exist_ok=True)
        with open(os.path.join(st_dir, "model.safetensors"), "wb") as f:
            f.write(b"weights")

        with self.assertRaises(ModelFormatError) as ctx:
            self.detector.detect(st_dir)
        self.assertIn(".safetensors weights without required config.json metadata", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 7. SpeechBrain VoxLingua107
    # -------------------------------------------------------------------------

    def test_speechbrain_voxlingua_format_and_dependency_rejection(self):
        """Verify SpeechBrain ECAPA_TDNN detection and DependencyMissingError rejection."""
        sb_dir = os.path.join(self.temp_dir, "voxlingua107")
        os.makedirs(sb_dir, exist_ok=True)
        with open(os.path.join(sb_dir, "hyperparams.yaml"), "w") as f:
            f.write("model_name: voxlingua107\n")
        with open(os.path.join(sb_dir, "embedding_model.ckpt"), "wb") as f:
            f.write(b"fake_embedding")
        with open(os.path.join(sb_dir, "classifier.ckpt"), "wb") as f:
            f.write(b"fake_classifier")

        fw, arch, mtype = self.detector.detect(sb_dir)
        self.assertEqual(fw, "speechbrain")
        self.assertEqual(arch, "ECAPA_TDNN")
        self.assertEqual(mtype, "voxlingua107")

        task = self.detector.detect_task(sb_dir, architecture=arch, model_type=mtype)
        self.assertEqual(task, "language_detection")

        loader = ModelLoader()
        with self.assertRaises(DependencyMissingError) as ctx:
            loader.validate_backend_compatibility(sb_dir, backend="python")
        self.assertIn("speechbrain", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 8. Faster Whisper & Whisper Variants
    # -------------------------------------------------------------------------

    def test_faster_whisper_format_and_dependency_rejection(self):
        """Verify CTranslate2 Faster Whisper detection and DependencyMissingError rejection."""
        whisper_dir = os.path.join(self.temp_dir, "faster_whisper_large_v3")
        os.makedirs(whisper_dir, exist_ok=True)
        with open(os.path.join(whisper_dir, "model.bin"), "wb") as f:
            f.write(b"ct2_weights")
        with open(os.path.join(whisper_dir, "vocabulary.json"), "w") as f:
            json.dump({"a": 1}, f)
        with open(os.path.join(whisper_dir, "config.json"), "w") as f:
            json.dump({"qformer": False}, f)

        fw, arch, mtype = self.detector.detect(whisper_dir)
        self.assertEqual(fw, "ctranslate2")
        self.assertEqual(arch, "WhisperModel")
        self.assertEqual(mtype, "faster_whisper")

        task = self.detector.detect_task(whisper_dir, architecture=arch, model_type=mtype)
        self.assertEqual(task, "transcription")

        loader = ModelLoader()
        with self.assertRaises(DependencyMissingError) as ctx:
            loader.validate_backend_compatibility(whisper_dir, backend="python")
        self.assertIn("ctranslate2", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 9. DeepFilterNet3 (Denoising & Directory Preservation)
    # -------------------------------------------------------------------------

    def test_deepfilternet3_format_and_dependency_rejection(self):
        """Verify DeepFilterNet3 detection, relative subdirectory preservation, and DependencyMissingError."""
        df_dir = os.path.join(self.temp_dir, "DeepFilterNet3")
        ckpts_dir = os.path.join(df_dir, "checkpoints")
        os.makedirs(ckpts_dir, exist_ok=True)
        with open(os.path.join(df_dir, "config.ini"), "w") as f:
            f.write("[deepfilternet3]\nsampling_rate=48000\n")
        with open(os.path.join(ckpts_dir, "model_120.ckpt.best"), "wb") as f:
            f.write(b"df_checkpoint_data")

        # Verify relative directory structure
        self.assertTrue(os.path.exists(os.path.join(df_dir, "checkpoints", "model_120.ckpt.best")))

        fw, arch, mtype = self.detector.detect(df_dir)
        self.assertEqual(fw, "deepfilternet")
        self.assertEqual(arch, "DeepFilterNet3")
        self.assertEqual(mtype, "deepfilternet3")

        task = self.detector.detect_task(df_dir, architecture=arch, model_type=mtype)
        self.assertEqual(task, "denoising")

        loader = ModelLoader()
        with self.assertRaises(DependencyMissingError) as ctx:
            loader.validate_backend_compatibility(df_dir, backend="python")
        self.assertIn("deepfilternet", str(ctx.exception))

    def test_deepfilternet3_tensor_contracts(self):
        """Verify denoising tensor contract: noisy audio in -> enhanced audio out."""
        mapper = TensorMapper(task="denoising")
        noisy_samples = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)

        inputs, outputs = mapper.build_request(noisy_samples, model_name="DeepFilterNet3")
        self.assertEqual(inputs[0]["name"], "AUDIO_SIGNAL")
        self.assertEqual(outputs, [{"name": "ENHANCED_AUDIO"}])

        mock_triton_response = {
            "model_name": "DeepFilterNet3",
            "outputs": [
                {"name": "ENHANCED_AUDIO", "datatype": "FP32", "shape": [16000], "data": noisy_samples.tolist()},
                {"name": "SAMPLE_RATE", "datatype": "INT32", "shape": [1], "data": [48000]},
            ]
        }
        res = mapper.parse_response(mock_triton_response, task="denoising")
        self.assertEqual(res["task"], "denoising")
        self.assertTrue(isinstance(res["enhanced_audio"], np.ndarray))
        self.assertEqual(res["sample_rate"], 48000)

    # -------------------------------------------------------------------------
    # 10. Model Switching via Configuration
    # -------------------------------------------------------------------------

    def test_model_switching_via_reload(self):
        """Verify dynamic model switching without modifying source code or restarting framework."""
        # 1. Setup Model A and Model B in temp files
        model_a = os.path.join(self.temp_dir, "stt_conformer.nemo")
        with open(model_a, "wb") as f:
            f.write(b"model_a_data")

        model_b = os.path.join(self.temp_dir, "ambernet.nemo")
        with open(model_b, "wb") as f:
            f.write(b"model_b_data")

        mock_engine = MagicMock(spec=TritonInferenceEngine)
        mock_engine.verify_readiness.return_value = True

        mock_loader = MagicMock(spec=ModelLoader)
        mock_loader.SUPPORTED_STORAGE_SCHEMES = {"s3", "minio"}
        mock_loader.load_for_triton.side_effect = lambda src, cfg, eng: {
            "model_name": os.path.splitext(os.path.basename(src))[0],
            "framework": "nemo",
            "model_source": src,
            "task": cfg.task,
        }

        cfg_a = FrameworkConfig(
            model_source=f"s3://speech-data/model/Transcription/{os.path.basename(model_a)}",
            task="transcription",
            triton_config=TritonConfig(model_name="conformer", check_model_ready=False),
        )

        mock_env = MagicMock()
        mock_env.validate_environment.return_value = True

        manager = ml.load(
            config=cfg_a,
            env_manager=mock_env,
            model_loader=mock_loader,
            inference_engine=mock_engine,
        )
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(manager.model_id, "stt_conformer")

        # 2. Switch configuration to Model B (AmberNet LID)
        manager.reload(
            new_source=f"s3://speech-data/model/Language_Detection/{os.path.basename(model_b)}",
            task="language_detection",
            model_name="ambernet",
        )
        self.assertEqual(manager.state, ExecutionState.READY)
        self.assertEqual(manager.model_id, "ambernet")
        self.assertEqual(manager.config.task, "language_detection")


if __name__ == "__main__":
    unittest.main()
