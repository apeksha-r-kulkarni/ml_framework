"""
Format Detector for ML Framework model artifacts.

Inspects local model directories and artifacts to identify framework,
architecture, and model type according to Sequence Diagram 2 in architecture.md.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional

from ..errors.exceptions import (
    ModelLoadError,
    ModelFormatError,
)

logger = logging.getLogger("ml_framework.models.detector")

# Known speech model types according to EnvironmentManager and Transformers specifications
SPEECH_MODEL_TYPES = {
    "whisper",
    "wav2vec2",
    "hubert",
    "speech_to_text",
    "speech2text",
    "mms",
    "unispeech",
    "unispeech_sat",
    "sew",
    "sew_d",
    "audio_spectrogram_transformer",
    "encodec",
    "seamless_m4t",
    "bark",
}


class FormatDetector:
    """
    Inspects model directory contents or artifact paths to detect:
    - framework: "speech", "transformers", "spacy", "onnx"
    - architecture: e.g. "WhisperForConditionalGeneration", "Wav2Vec2ForCTC", "BertForNER"
    - model_type: e.g. "whisper", "wav2vec2", "bert"
    """

    def detect(
        self,
        model_dir: str,
        configured_framework: str = "auto"
    ) -> Tuple[str, str, str]:
        """
        Detects framework, architecture, and model_type from a local directory or file.
        Raises ModelLoadError if path does not exist, or ModelFormatError if unrecognized.
        """
        if not model_dir or not str(model_dir).strip():
            raise ModelLoadError("Model path cannot be empty or None")

        clean_path = str(model_dir).strip()
        if clean_path.startswith("file://"):
            clean_path = clean_path[7:]

        if not os.path.exists(clean_path):
            raise ModelLoadError(
                f"Model path does not exist: '{clean_path}'",
                details={"model_dir": clean_path}
            )

        norm_cfg_fw = (configured_framework or "auto").lower().strip()

        # 1. Single file check (.onnx, .nemo, .safetensors)
        if os.path.isfile(clean_path):
            lower_name = clean_path.lower()
            if lower_name.endswith(".safetensors"):
                raise ModelFormatError(
                    f"Standalone .safetensors files cannot be loaded without model configuration and architecture metadata. "
                    f"A configuration file (config.json) is required: '{clean_path}'",
                    details={"model_path": clean_path, "file_type": "safetensors"}
                )
            if lower_name.endswith(".onnx"):
                if norm_cfg_fw not in ("auto", "onnx"):
                    raise ModelFormatError(
                        f"Configured framework '{configured_framework}' conflicts with standalone ONNX model '{clean_path}'."
                    )
                return ("onnx", "ONNXModel", os.path.splitext(os.path.basename(clean_path))[0])
            if lower_name.endswith(".nemo"):
                if norm_cfg_fw not in ("auto", "nemo", "speech"):
                    raise ModelFormatError(
                        f"Configured framework '{configured_framework}' conflicts with NeMo archive '{clean_path}'."
                    )
                base_name = os.path.splitext(os.path.basename(clean_path))[0]
                lower_base = base_name.lower()
                if any(k in lower_base for k in ("ambernet", "pearlnet", "speaker", "lang", "lid")):
                    return ("nemo", "EncDecSpeakerLabelModel", "nemo_lid")
                if "conformer" in lower_base:
                    return ("nemo", "EncDecCTCModelBPE", base_name)
                return ("nemo", "NeMoModel", base_name)
            raise ModelFormatError(
                f"Unsupported single-file model: '{clean_path}'. Expected directory, .onnx, or .nemo archive."
            )

        # 2. Directory inspection
        try:
            entries = os.listdir(clean_path)
        except OSError as e:
            raise ModelLoadError(f"Cannot list model directory '{clean_path}': {e}") from e

        if not entries:
            raise ModelFormatError(f"Model directory '{clean_path}' is empty.")

        has_onnx = any(f.lower().endswith(".onnx") for f in entries)
        has_nemo = any(f.lower().endswith(".nemo") for f in entries)
        has_safetensors = any(f.lower().endswith(".safetensors") for f in entries)
        config_path = os.path.join(clean_path, "config.json")
        meta_path = os.path.join(clean_path, "meta.json")
        cfg_path = os.path.join(clean_path, "config.cfg")

        # Reject directory containing only .safetensors without config.json
        if has_safetensors and not os.path.exists(config_path):
            raise ModelFormatError(
                f"Model directory '{clean_path}' contains .safetensors weights without required config.json metadata.",
                details={"model_dir": clean_path, "entries": entries}
            )

        # Check for DeepFilterNet3 format (config.ini or checkpoints/*.ckpt.best)
        df_ini = os.path.join(clean_path, "config.ini")
        has_df_checkpoints = os.path.isdir(os.path.join(clean_path, "checkpoints")) or any(
            f.endswith(".ckpt.best") for f in entries
        )
        if os.path.exists(df_ini) or (has_df_checkpoints and not os.path.exists(config_path)):
            if norm_cfg_fw not in ("auto", "deepfilternet", "df", "deepfilternet3"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with DeepFilterNet model at '{clean_path}'."
                )
            return ("deepfilternet", "DeepFilterNet3", "deepfilternet3")

        # Check for SpeechBrain format (hyperparams.yaml)
        sb_yaml = os.path.join(clean_path, "hyperparams.yaml")
        if os.path.exists(sb_yaml):
            if norm_cfg_fw not in ("auto", "speechbrain", "voxlingua107"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with SpeechBrain model at '{clean_path}'."
                )
            return ("speechbrain", "ECAPA_TDNN", "voxlingua107")

        # Check for CTranslate2 / Faster Whisper format (model.bin + vocabulary/tokenizer)
        has_ct2_bin = "model.bin" in entries and not any(
            f.endswith(".safetensors") or f == "pytorch_model.bin" for f in entries
        )
        has_ct2_vocab = any(f in entries for f in ("vocabulary.json", "vocabulary.txt", "assets.json"))
        if has_ct2_bin and has_ct2_vocab:
            if norm_cfg_fw not in ("auto", "ctranslate2", "faster_whisper", "whisper", "speech"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with CTranslate2 / Faster Whisper model at '{clean_path}'."
                )
            model_type = "faster_whisper" if "faster" in clean_path.lower() else "whisper"
            return ("ctranslate2", "WhisperModel", model_type)

        # Check for HuggingFace / Transformers format (config.json)
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception as e:
                raise ModelFormatError(
                    f"Corrupted or invalid config.json in '{clean_path}': {e}",
                    details={"model_dir": clean_path, "error": str(e)}
                ) from e

            if not isinstance(cfg_data, dict):
                raise ModelFormatError(
                    f"config.json in '{clean_path}' must be a JSON object, got {type(cfg_data).__name__}"
                )

            model_type = str(cfg_data.get("model_type", "") or "").lower()
            raw_archs = cfg_data.get("architectures") or []
            archs = [str(a) for a in raw_archs] if isinstance(raw_archs, (list, tuple)) else []
            primary_arch = archs[0] if archs else "Model"

            # Check if speech model
            is_speech = (
                model_type in SPEECH_MODEL_TYPES
                or any(
                    any(s in a.lower() for s in ("whisper", "wav2vec", "hubert", "speech"))
                    for a in archs
                )
            )

            # Check if Speech Sequence Classification (e.g. MMS Language Detection)
            if is_speech and (any("sequenceclassification" in a.lower() for a in archs) or "langs.txt" in entries):
                if norm_cfg_fw not in ("auto", "transformers", "huggingface", "speech"):
                    raise ModelFormatError(
                        f"Configured framework is '{configured_framework}', but model at '{clean_path}' is a Transformers SequenceClassification model."
                    )
                return ("transformers", primary_arch or "Wav2Vec2ForSequenceClassification", "wav2vec2_seq_class")

            # Also check preprocessor_config.json for speech feature extractor
            preproc_path = os.path.join(clean_path, "preprocessor_config.json")
            if os.path.exists(preproc_path):
                try:
                    with open(preproc_path, "r", encoding="utf-8") as f:
                        pre_data = json.load(f)
                    if isinstance(pre_data, dict):
                        fe_type = str(pre_data.get("feature_extractor_type", "")).lower()
                        if any(s in fe_type for s in ("wav2vec", "whisper", "speech", "sequence")):
                            is_speech = True
                except Exception:
                    pass

            if is_speech:
                if norm_cfg_fw not in ("auto", "speech", "transformers", "huggingface"):
                    raise ModelFormatError(
                        f"Configured framework is '{configured_framework}', but model at '{clean_path}' is a speech model."
                    )
                return ("speech", primary_arch, model_type or "speech")

            if norm_cfg_fw == "speech":
                raise ModelFormatError(
                    f"Configured framework is 'speech', but model at '{clean_path}' is of type '{model_type}' (architecture: {primary_arch})."
                )

            if has_onnx:
                if norm_cfg_fw not in ("auto", "onnx", "onnxruntime"):
                    raise ModelFormatError(
                        f"Configured framework is '{configured_framework}', but model at '{clean_path}' contains ONNX models."
                    )
                return ("onnx", primary_arch, model_type)

            if norm_cfg_fw not in ("auto", "transformers", "huggingface"):
                raise ModelFormatError(
                    f"Configured framework is '{configured_framework}', but model at '{clean_path}' is a Transformers model."
                )

            return ("transformers", primary_arch, model_type)

        # Check for spaCy format (meta.json or config.cfg)
        if os.path.exists(meta_path) or os.path.exists(cfg_path):
            if norm_cfg_fw not in ("auto", "spacy"):
                raise ModelFormatError(
                    f"Configured framework is '{configured_framework}', but model at '{clean_path}' is a spaCy pipeline (found meta.json/config.cfg)."
                )
            pipeline_name = "spacy_pipeline"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    pipeline_name = meta_data.get("name", "spacy_pipeline")
                except Exception:
                    pass
            return ("spacy", "spaCyPipeline", pipeline_name)

        # Check for ONNX directory (contains .onnx files and no config.json)
        if has_onnx:
            if norm_cfg_fw not in ("auto", "onnx", "onnxruntime"):
                raise ModelFormatError(
                    f"Configured framework is '{configured_framework}', but model at '{clean_path}' contains ONNX models without speech configuration."
                )
            return ("onnx", "ONNXModel", "onnx")

        # Check for NeMo directory (contains model_config.yaml)
        nemo_yaml_path = os.path.join(clean_path, "model_config.yaml")
        if os.path.exists(nemo_yaml_path):
            if norm_cfg_fw not in ("auto", "nemo", "speech"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with detected NeMo model at '{clean_path}'."
                )
            arch = "EncDecCTCModelBPE"
            m_type = "conformer_ctc"
            try:
                import yaml
                with open(nemo_yaml_path, "r", encoding="utf-8") as yf:
                    ydata = yaml.safe_load(yf)
                if isinstance(ydata, dict):
                    target = str(ydata.get("target", "") or "")
                    encoder_target = str(ydata.get("encoder", {}).get("_target_", "") or "") if isinstance(ydata.get("encoder"), dict) else ""
                    if target:
                        arch = target.split(".")[-1]
                    elif encoder_target:
                        arch = encoder_target.split(".")[-1]
                    if any(k in target.lower() for k in ("speaker", "label", "classification")):
                        arch = "EncDecSpeakerLabelModel"
                        m_type = "nemo_lid"
                    elif "conformer" in target.lower() or "conformer" in encoder_target.lower():
                        m_type = "conformer_ctc"
                    else:
                        m_type = "nemo_asr"
            except Exception:
                pass
            return ("nemo", arch, m_type)

        # Check for directory containing .nemo archive
        if has_nemo:
            if norm_cfg_fw not in ("auto", "nemo", "speech"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with detected NeMo model archive in '{clean_path}'."
                )
            nemo_file = next(f for f in entries if f.lower().endswith(".nemo"))
            model_name = os.path.splitext(nemo_file)[0]
            lower_name = model_name.lower()
            if any(k in lower_name for k in ("ambernet", "pearlnet", "speaker", "lang", "lid")):
                arch = "EncDecSpeakerLabelModel"
                m_type = "nemo_lid"
            elif "conformer" in lower_name:
                arch = "EncDecCTCModelBPE"
                m_type = "conformer_ctc"
            else:
                arch = "NeMoModel"
                m_type = "nemo_asr"
            return ("nemo", arch, m_type)

        # Check for Triton Model Repository (contains config.pbtxt)
        triton_cfg_path = os.path.join(clean_path, "config.pbtxt")
        if os.path.exists(triton_cfg_path):
            if norm_cfg_fw not in ("auto", "triton"):
                raise ModelFormatError(
                    f"Configured framework '{configured_framework}' conflicts with Triton model repository '{clean_path}'."
                )
            return ("triton", "TritonModel", os.path.basename(clean_path))

        raise ModelFormatError(
            f"No recognized model configuration found in '{clean_path}'. "
            f"Expected config.json (Transformers/Speech), meta.json/config.cfg (spaCy), model_config.yaml (NeMo), "
            f"config.ini (DeepFilterNet), hyperparams.yaml (SpeechBrain), config.pbtxt (Triton), or .onnx/.nemo file."
        )

    def detect_task(
        self,
        model_path: str,
        architecture: str = "",
        model_type: str = "",
        configured_task: str = "auto",
    ) -> str:
        """
        Determines the inference task ('transcription', 'language_detection', 'denoising')
        from model path, architecture, and model_type, or honors valid configured_task override.
        """
        if configured_task and configured_task != "auto":
            norm = configured_task.lower().strip()
            valid_tasks = {"transcription", "language_detection", "denoising"}
            if norm not in valid_tasks:
                raise ModelFormatError(
                    f"Unsupported task '{configured_task}'. Supported tasks: {sorted(list(valid_tasks))}"
                )
            return norm

        lower_path = str(model_path).lower()
        lower_arch = str(architecture).lower()
        lower_type = str(model_type).lower()

        # 1. Denoising
        if "denois" in lower_path or "deepfilter" in lower_arch or "deepfilter" in lower_type:
            return "denoising"

        # 2. Language Detection
        if (
            "language_detection" in lower_path
            or "speakerlabel" in lower_arch
            or "sequenceclassification" in lower_arch
            or "lid" in lower_type
            or "voxlingua" in lower_type
            or "ambernet" in lower_path
            or "pearlnet" in lower_path
        ):
            return "language_detection"

        # 3. Default: Transcription
        return "transcription"
