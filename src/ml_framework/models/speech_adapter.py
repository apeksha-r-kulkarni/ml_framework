"""
Speech Model Framework Adapter for HuggingFace/PyTorch Speech Models.

Encapsulates offline loading, device placement, audio preprocessing,
and inference for Whisper, Wav2Vec2, MMS, HuBERT, and related speech model architectures.
Supports speech-to-text (CTC & Seq2Seq) and audio/language identification tasks.
"""

import os
import gc
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..interfaces.base import BaseModelAdapter
from ..config.model_config import FrameworkConfig
from ..errors.exceptions import (
    ModelLoadError,
    DependencyMissingError,
    HardwareIncompatibleError,
    InferenceError,
    InputProcessingError,
)
from ..input.audio import decode_and_resample_audio

logger = logging.getLogger("ml_framework.models.speech")


class SpeechModelAdapter(BaseModelAdapter):
    """
    Framework adapter for speech models.
    Encapsulates the loaded PyTorch model and audio processor/feature extractor.
    Enforces strictly local, offline loading (local_files_only=True) and executes
    speech-to-text and audio/language classification inference.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        architecture: str = "SpeechModel",
        model_type: str = "speech",
        config: Optional[FrameworkConfig] = None,
        model: Optional[Any] = None,
        processor: Optional[Any] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.architecture = architecture
        self.model_type = model_type
        self.config = config
        self.model = model
        self.processor = processor
        self.hf_config: Optional[Any] = None
        self.task: str = "speech_to_text"
        self.sampling_rate: int = 16000
        self._is_loaded = model is not None
        self._is_nemo = False

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and active in memory."""
        return self._is_loaded and self.model is not None

    def _is_nemo_target(self) -> bool:
        """Determines if the current model target is a NeMo model."""
        if getattr(self, "_is_nemo", False):
            return True
        if str(self.model_type).lower() in ("conformer_ctc", "nemo_asr", "nemo"):
            return True
        if str(self.architecture).lower() in ("encdecctcmodelbpe", "nemomodel"):
            return True
        if isinstance(self.model_dir, str):
            clean = str(self.model_dir).strip()
            if clean.lower().endswith(".nemo"):
                return True
            if os.path.isdir(clean):
                if os.path.exists(os.path.join(clean, "model_config.yaml")):
                    return True
                try:
                    if any(f.lower().endswith(".nemo") for f in os.listdir(clean)):
                        return True
                except Exception:
                    pass
        if self.config and getattr(self.config, "framework", "").lower() == "nemo":
            return True
        return False

    def _load_nemo(self) -> "SpeechModelAdapter":
        """
        Loads NVIDIA NeMo speech recognition model checkpoint offline without ONNX.
        Supports single .nemo archive files as well as extracted NeMo model directories.
        """
        try:
            import torch
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            missing = []
            err_str = str(e).lower()
            if "torch" in err_str:
                missing.append("torch")
            if "nemo" in err_str:
                missing.append("nemo_toolkit")
            if not missing:
                missing = ["torch", "nemo_toolkit"]
            raise DependencyMissingError(
                f"Required dependencies for NeMo speech model loading are missing: {e}. "
                f"Install via: pip install '.[speech]' or pip install nemo_toolkit soundfile numpy torch",
                missing_packages=missing,
                details={"error": str(e), "missing_packages": missing}
            ) from e

        clean_path = str(self.model_dir).strip()
        restore_path = clean_path
        if os.path.isdir(clean_path):
            nemo_files = [f for f in os.listdir(clean_path) if f.lower().endswith(".nemo")]
            if nemo_files:
                restore_path = os.path.join(clean_path, nemo_files[0])
            elif not os.path.exists(os.path.join(clean_path, "model_config.yaml")):
                raise ModelLoadError(
                    f"Invalid NeMo model directory '{clean_path}': missing .nemo file or model_config.yaml",
                    details={"model_dir": clean_path}
                )

        try:
            target_map_location = "cuda" if "cuda" in str(self.device).lower() and torch.cuda.is_available() else "cpu"
            if hasattr(nemo_asr.models, "EncDecCTCModelBPE"):
                model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
                    restore_path=restore_path,
                    map_location=target_map_location,
                )
            elif hasattr(nemo_asr.models, "ASRModel"):
                model = nemo_asr.models.ASRModel.restore_from(
                    restore_path=restore_path,
                    map_location=target_map_location,
                )
            else:
                raise ModelLoadError(
                    f"No suitable NeMo ASR loader found in nemo.collections.asr for '{restore_path}'"
                )
        except Exception as load_err:
            if isinstance(load_err, (DependencyMissingError, HardwareIncompatibleError, ModelLoadError)):
                raise
            raise ModelLoadError(
                f"Failed to load NeMo speech model weights from '{restore_path}' (arch={self.architecture}): {load_err}.",
                details={
                    "model_dir": self.model_dir,
                    "restore_path": restore_path,
                    "architecture": self.architecture,
                    "error": str(load_err),
                }
            ) from load_err

        # Resolve target sampling rate
        sr = getattr(getattr(model, "preprocessor", None), "_sample_rate", 16000) or getattr(model, "sampling_rate", 16000)
        self.sampling_rate = int(sr)
        self.task = "speech_to_text"
        if not self.architecture or self.architecture == "SpeechModel":
            self.architecture = type(model).__name__

        # Place model on device and set evaluation mode
        try:
            if hasattr(model, "to") and callable(model.to):
                model.to(self.device)
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()
        except Exception as dev_err:
            raise HardwareIncompatibleError(
                f"Failed to move speech model to device '{self.device}': {dev_err}",
                details={"device": self.device, "error": str(dev_err)}
            ) from dev_err

        self.model = model
        self.processor = None
        self.hf_config = None
        self._is_loaded = True
        self._is_nemo = True
        logger.info(
            f"NeMo speech model loaded successfully from '{restore_path}' "
            f"(arch={self.architecture}, task={self.task}, device={self.device}, sr={self.sampling_rate}Hz)."
        )
        return self

    def load(self) -> "SpeechModelAdapter":
        """
        Loads speech model weights and processor from local files only.
        Guarantees strictly offline execution (local_files_only=True).
        Identifies task type (speech-to-text vs audio/language classification)
        from configuration metadata.
        """
        if self._is_loaded and self.model is not None:
            return self

        if self._is_nemo_target():
            return self._load_nemo()

        try:
            import torch
            from transformers import AutoConfig
        except ImportError as e:
            missing = []
            err_str = str(e).lower()
            if "torch" in err_str:
                missing.append("torch")
            if "transformers" in err_str:
                missing.append("transformers")
            if not missing:
                missing = ["torch", "transformers"]
            raise DependencyMissingError(
                f"Required dependencies for speech model loading are missing: {e}. "
                f"Install via: pip install '.[speech]' or pip install torch transformers soundfile numpy",
                missing_packages=missing,
                details={"error": str(e), "missing_packages": missing}
            ) from e

        # Load HuggingFace config offline
        try:
            hf_config = AutoConfig.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
        except Exception as cfg_err:
            raise ModelLoadError(
                f"Failed to load speech model configuration from '{self.model_dir}': {cfg_err}. "
                f"Ensure 'config.json' is present and valid.",
                details={"model_dir": self.model_dir, "error": str(cfg_err)}
            ) from cfg_err

        # Inspect architectures and task type
        raw_archs = getattr(hf_config, "architectures", []) or []
        archs = [str(a) for a in raw_archs] if isinstance(raw_archs, (list, tuple)) else []
        if archs and (not self.architecture or self.architecture == "SpeechModel"):
            self.architecture = archs[0]

        arch_str = " ".join([a.lower() for a in archs] + [self.architecture.lower()])

        is_classification = any(
            k in arch_str for k in ("sequenceclassification", "audioclassification", "speaker", "lid")
        )
        is_seq2seq = (
            getattr(hf_config, "is_encoder_decoder", False)
            or "whisper" in self.model_type.lower()
            or "whisper" in getattr(hf_config, "model_type", "").lower()
            or any("conditionalgeneration" in a.lower() for a in archs)
        )

        model = None
        try:
            if is_classification:
                try:
                    from transformers import AutoModelForAudioClassification
                    model = AutoModelForAudioClassification.from_pretrained(
                        self.model_dir,
                        config=hf_config,
                        local_files_only=True,
                    )
                except Exception:
                    try:
                        from transformers import Wav2Vec2ForSequenceClassification
                        model = Wav2Vec2ForSequenceClassification.from_pretrained(
                            self.model_dir,
                            config=hf_config,
                            local_files_only=True,
                        )
                    except Exception:
                        from transformers import AutoModelForSequenceClassification
                        model = AutoModelForSequenceClassification.from_pretrained(
                            self.model_dir,
                            config=hf_config,
                            local_files_only=True,
                        )
                self.task = "language_identification"
            elif is_seq2seq:
                from transformers import AutoModelForSpeechSeq2Seq
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
                self.task = "speech_seq2seq"
            else:
                from transformers import AutoModelForCTC
                model = AutoModelForCTC.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
                self.task = "speech_to_text"
        except Exception as load_err:
            raise ModelLoadError(
                f"Failed to load speech model weights from '{self.model_dir}' (arch={self.architecture}): {load_err}. "
                f"Ensure weight files (model.safetensors or pytorch_model.bin) are present locally.",
                details={
                    "model_dir": self.model_dir,
                    "architecture": self.architecture,
                    "error": str(load_err),
                }
            ) from load_err

        # Load processor or feature extractor
        processor = None
        try:
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        except Exception:
            try:
                from transformers import AutoFeatureExtractor
                processor = AutoFeatureExtractor.from_pretrained(self.model_dir, local_files_only=True)
            except Exception:
                logger.info(f"No processor or feature extractor loaded for speech model at '{self.model_dir}'")

        # Resolve target sampling rate
        sr = 16000
        if processor is not None:
            sr = (
                getattr(processor, "sampling_rate", None)
                or getattr(getattr(processor, "feature_extractor", None), "sampling_rate", None)
                or 16000
            )
        elif hasattr(hf_config, "sampling_rate") and hf_config.sampling_rate:
            sr = int(hf_config.sampling_rate)
        self.sampling_rate = int(sr)

        # Place model on device
        try:
            if hasattr(model, "to") and callable(model.to):
                model.to(self.device)
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()
        except Exception as dev_err:
            raise HardwareIncompatibleError(
                f"Failed to move speech model to device '{self.device}': {dev_err}",
                details={"device": self.device, "error": str(dev_err)}
            ) from dev_err

        self.model = model
        self.processor = processor
        self.hf_config = hf_config
        self._is_loaded = True
        logger.info(
            f"Speech model loaded successfully from '{self.model_dir}' "
            f"(arch={self.architecture}, task={self.task}, device={self.device}, sr={self.sampling_rate}Hz)."
        )
        return self

    def unload(self) -> None:
        """Safely unloads model instances and frees execution memory."""
        if self.model is not None:
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    self.model.to("cpu")
            except Exception:
                pass
            self.model = None

        self.processor = None
        self.hf_config = None
        self._is_loaded = False
        self._is_nemo = False

        # Clear CUDA memory if PyTorch CUDA was active
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info(f"Speech model unloaded from memory (model_dir='{self.model_dir}').")

    def predict(self, preprocessed_input: Any) -> Any:
        """
        Executes actual speech inference on preprocessed_input using PyTorch inference mode.
        Accepts raw audio bytes (WAV, MP3, FLAC, OGG), file paths, or pre-decoded waveforms.
        Routes to task-specific inference handler (speech-to-text CTC, classification, or seq2seq).
        """
        if not self.is_loaded or self.model is None:
            raise ModelLoadError("Cannot execute prediction on unloaded speech model.")

        if preprocessed_input is None:
            raise InferenceError("Inference input cannot be None.")

        try:
            import torch
        except ImportError as e:
            raise DependencyMissingError(
                f"PyTorch ('torch') is required for speech inference: {e}",
                missing_packages=["torch"],
            ) from e

        # 1. Decode and resample audio input to target model sampling rate
        try:
            waveform, duration_sec = decode_and_resample_audio(
                preprocessed_input,
                target_sampling_rate=self.sampling_rate,
            )
        except (InputProcessingError, DependencyMissingError):
            raise
        except Exception as e:
            raise InputProcessingError(f"Failed to decode audio input for speech model: {e}") from e

        # Route to NeMo CTC predictor if targeting NeMo model
        if self._is_nemo_target():
            return self._predict_nemo_ctc(waveform, duration_sec)

        # 2. Extract features / tokenize using processor or feature extractor
        if self.processor is not None:
            try:
                inputs = self.processor(
                    waveform,
                    sampling_rate=self.sampling_rate,
                    return_tensors="pt",
                )
            except Exception as proc_err:
                raise InferenceError(
                    f"Speech processor failed to extract features from audio: {proc_err}",
                    details={"sampling_rate": self.sampling_rate, "error": str(proc_err)}
                ) from proc_err
        else:
            inputs = {"input_values": torch.from_numpy(waveform).unsqueeze(0)}

        # Move tensors to model device
        model_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                model_inputs[k] = v.to(self.device)
            else:
                model_inputs[k] = v

        # 3. Route to task-specific handler
        if self.task == "language_identification":
            return self._predict_language_identification(model_inputs, duration_sec)
        elif self.task == "speech_seq2seq":
            return self._predict_seq2seq(model_inputs, duration_sec)
        else:
            return self._predict_speech_to_text_ctc(model_inputs, duration_sec)

    def _predict_nemo_ctc(self, waveform: Any, duration_sec: float) -> Dict[str, Any]:
        """Runs forward pass and CTC greedy decoding for NeMo ASR models."""
        import torch

        # Check for silent audio (all zeros or below threshold)
        max_val = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
        if len(waveform) == 0 or max_val < 1e-6:
            return {
                "task": "speech_to_text",
                "architecture": self.architecture,
                "model_type": self.model_type,
                "text": "",
                "transcription": "",
                "confidence": 1.0,
                "sampling_rate": self.sampling_rate,
                "duration_seconds": round(float(duration_sec), 3),
            }

        audio_tensor = torch.from_numpy(waveform.astype(np.float32))
        if max_val > 1.0:
            audio_tensor = audio_tensor / 32768.0

        audio_signal = audio_tensor.unsqueeze(0).to(self.device)
        audio_length = torch.tensor([audio_tensor.shape[0]], dtype=torch.long, device=self.device)

        try:
            with torch.inference_mode():
                outputs = self.model(input_signal=audio_signal, input_signal_length=audio_length)
                if isinstance(outputs, tuple):
                    logprobs = outputs[0]
                    encoded_len = outputs[1] if len(outputs) > 1 else None
                elif hasattr(outputs, "logits"):
                    logprobs = outputs.logits
                    encoded_len = getattr(outputs, "logits_length", None)
                else:
                    logprobs = outputs
                    encoded_len = None

                transcription = ""
                if hasattr(self.model, "decoding") and hasattr(self.model.decoding, "ctc_decoder_predictions_tensor"):
                    hypotheses = self.model.decoding.ctc_decoder_predictions_tensor(
                        logprobs, decoder_lengths=encoded_len
                    )
                    if hypotheses and hypotheses[0]:
                        hyp = hypotheses[0][0]
                        transcription = hyp.text if hasattr(hyp, "text") else str(hyp)
                elif hasattr(self.model, "transcribe"):
                    transcription_list = self.model.transcribe([waveform])
                    transcription = transcription_list[0] if transcription_list else ""
                else:
                    predicted_ids = torch.argmax(logprobs, dim=-1)
                    if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "ids_to_text"):
                        transcription = self.model.tokenizer.ids_to_text(predicted_ids[0].tolist())
                    else:
                        raise InferenceError("NeMo model does not provide a supported CTC decoding method.")
        except Exception as infer_err:
            if isinstance(infer_err, InferenceError):
                raise
            raise InferenceError(
                f"NeMo speech-to-text forward pass failed: {infer_err}",
                details={"architecture": self.architecture, "error": str(infer_err)}
            ) from infer_err

        transcription = str(transcription).strip()

        # Compute average token confidence score
        try:
            probs = torch.softmax(logprobs, dim=-1)
            max_probs = torch.max(probs, dim=-1).values[0]
            confidence = float(torch.mean(max_probs).item()) if max_probs.numel() > 0 else 1.0
        except Exception:
            confidence = 1.0

        return {
            "task": "speech_to_text",
            "architecture": self.architecture,
            "model_type": self.model_type,
            "text": transcription,
            "transcription": transcription,
            "confidence": round(float(confidence), 4),
            "sampling_rate": self.sampling_rate,
            "duration_seconds": round(float(duration_sec), 3),
        }

    def _predict_speech_to_text_ctc(self, inputs: Dict[str, Any], duration_sec: float) -> Dict[str, Any]:
        """Runs forward pass and CTC greedy decoding for speech transcription."""
        import torch

        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
        except Exception as infer_err:
            raise InferenceError(
                f"Speech-to-text forward pass failed: {infer_err}",
                details={"architecture": self.architecture, "error": str(infer_err)}
            ) from infer_err

        if not hasattr(outputs, "logits"):
            raise InferenceError(
                f"Model output does not contain expected 'logits' attribute. Got: {type(outputs).__name__}"
            )

        logits = outputs.logits
        if logits is None or logits.ndim < 2:
            raise InferenceError(f"Model returned invalid or empty logits: {logits}")

        predicted_ids = torch.argmax(logits, dim=-1)

        # Decode tokens to string using processor/tokenizer
        transcription = ""
        try:
            if hasattr(self.processor, "batch_decode"):
                decoded = self.processor.batch_decode(predicted_ids)
                transcription = decoded[0] if decoded else ""
            elif hasattr(self.processor, "decode"):
                transcription = self.processor.decode(predicted_ids[0])
            elif hasattr(self.processor, "tokenizer") and hasattr(self.processor.tokenizer, "batch_decode"):
                decoded = self.processor.tokenizer.batch_decode(predicted_ids)
                transcription = decoded[0] if decoded else ""
            else:
                raise InferenceError("Processor does not support CTC token decoding.")
        except Exception as dec_err:
            if isinstance(dec_err, InferenceError):
                raise
            raise InferenceError(f"Failed to decode predicted tokens to text: {dec_err}") from dec_err

        transcription = str(transcription).strip()

        # Compute average token confidence score
        try:
            probs = torch.softmax(logits, dim=-1)
            max_probs = torch.max(probs, dim=-1).values[0]
            confidence = float(torch.mean(max_probs).item()) if max_probs.numel() > 0 else 1.0
        except Exception:
            confidence = 1.0

        return {
            "task": "speech_to_text",
            "architecture": self.architecture,
            "model_type": self.model_type,
            "text": transcription,
            "transcription": transcription,
            "confidence": round(float(confidence), 4),
            "sampling_rate": self.sampling_rate,
            "duration_seconds": round(float(duration_sec), 3),
        }

    def _predict_language_identification(self, inputs: Dict[str, Any], duration_sec: float) -> Dict[str, Any]:
        """Runs forward pass and classification label resolution for MMS/audio classification."""
        import torch

        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
        except Exception as infer_err:
            raise InferenceError(
                f"Language identification forward pass failed: {infer_err}",
                details={"architecture": self.architecture, "error": str(infer_err)}
            ) from infer_err

        if not hasattr(outputs, "logits"):
            raise InferenceError(
                f"Classification model output does not contain expected 'logits' attribute. Got: {type(outputs).__name__}"
            )

        logits = outputs.logits
        if logits is None or logits.ndim < 2:
            raise InferenceError(f"Model returned invalid or empty logits: {logits}")

        probs = torch.softmax(logits, dim=-1)[0]
        pred_class_id = int(torch.argmax(probs).item())
        confidence = float(probs[pred_class_id].item())

        # Retrieve label mapping from config metadata
        id2label = self._get_id2label()
        predicted_label = (
            id2label.get(pred_class_id)
            or id2label.get(str(pred_class_id))
            or f"LABEL_{pred_class_id}"
        )

        # Extract top 5 predictions with scores
        top_k = min(5, len(probs))
        top_values, top_indices = torch.topk(probs, k=top_k)
        top_predictions = [
            {
                "language": id2label.get(int(idx.item())) or id2label.get(str(int(idx.item()))) or f"LABEL_{int(idx.item())}",
                "class_id": int(idx.item()),
                "score": round(float(val.item()), 4),
            }
            for val, idx in zip(top_values, top_indices)
        ]

        return {
            "task": "language_identification",
            "architecture": self.architecture,
            "model_type": self.model_type,
            "language": predicted_label,
            "label": predicted_label,
            "class_id": pred_class_id,
            "confidence": round(float(confidence), 4),
            "score": round(float(confidence), 4),
            "top_predictions": top_predictions,
            "sampling_rate": self.sampling_rate,
            "duration_seconds": round(float(duration_sec), 3),
        }

    def _predict_seq2seq(self, inputs: Dict[str, Any], duration_sec: float) -> Dict[str, Any]:
        """Runs autoregressive generation for sequence-to-sequence speech models (e.g. Whisper)."""
        import torch

        try:
            gen_inputs = inputs.get("input_features", inputs.get("input_values"))
            with torch.inference_mode():
                if hasattr(self.model, "generate"):
                    predicted_ids = self.model.generate(gen_inputs)
                else:
                    outputs = self.model(**inputs)
                    predicted_ids = torch.argmax(outputs.logits, dim=-1)
        except Exception as infer_err:
            raise InferenceError(
                f"Seq2Seq speech forward/generate pass failed: {infer_err}",
                details={"architecture": self.architecture, "error": str(infer_err)}
            ) from infer_err

        transcription = ""
        try:
            if hasattr(self.processor, "batch_decode"):
                decoded = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                transcription = decoded[0] if decoded else ""
            elif hasattr(self.processor, "decode"):
                transcription = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
        except Exception as dec_err:
            raise InferenceError(f"Failed to decode generated speech tokens: {dec_err}") from dec_err

        transcription = str(transcription).strip()
        return {
            "task": "speech_seq2seq",
            "architecture": self.architecture,
            "model_type": self.model_type,
            "text": transcription,
            "transcription": transcription,
            "confidence": 1.0,
            "sampling_rate": self.sampling_rate,
            "duration_seconds": round(float(duration_sec), 3),
        }

    def _get_id2label(self) -> Dict[Any, str]:
        """Retrieves id2label mapping from model config, hf_config, or langs.txt."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label") and self.model.config.id2label:
            return self.model.config.id2label
        if self.hf_config and hasattr(self.hf_config, "id2label") and self.hf_config.id2label:
            return self.hf_config.id2label
        langs_path = os.path.join(self.model_dir, "langs.txt")
        if os.path.exists(langs_path):
            try:
                with open(langs_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                return {idx: lang for idx, lang in enumerate(lines)}
            except Exception:
                pass
        return {}
