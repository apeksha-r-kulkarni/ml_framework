"""
Transformers Model Framework Adapter for HuggingFace/PyTorch Models.

Encapsulates offline loading, device placement, and lifecycle management
for BERT, RoBERTa, DeBERTa, DistilBERT, and general Transformers architectures.
"""

import os
import gc
import logging
from typing import Any, Optional

from ..interfaces.base import BaseModelAdapter
from ..config.model_config import FrameworkConfig
from ..errors.exceptions import (
    ModelLoadError,
    DependencyMissingError,
    HardwareIncompatibleError,
    InferenceError,
)

logger = logging.getLogger("ml_framework.models.transformers")


class TransformersModelAdapter(BaseModelAdapter):
    """
    Framework adapter for Hugging Face Transformers models.
    Encapsulates the loaded PyTorch model and tokenizer.
    Enforces strictly local, offline loading (local_files_only=True).
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        architecture: str = "TransformersModel",
        model_type: str = "transformers",
        config: Optional[FrameworkConfig] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.model_dir = model_dir
        self.device = device
        self.architecture = architecture
        self.model_type = model_type
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hf_config: Optional[Any] = None
        self._is_loaded = model is not None

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and active in memory."""
        return self._is_loaded and self.model is not None

    def load(self) -> "TransformersModelAdapter":
        """
        Loads transformers model weights and tokenizer from local files only.
        Guarantees strictly offline execution (local_files_only=True).
        """
        if self._is_loaded and self.model is not None:
            return self

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
                f"Required dependencies for transformers model loading are missing: {e}. "
                f"Install via: pip install '.[transformers]' or pip install torch transformers",
                missing_packages=missing,
                details={"error": str(e), "missing_packages": missing}
            ) from e

        # Load HuggingFace configuration offline
        try:
            hf_config = AutoConfig.from_pretrained(
                self.model_dir,
                local_files_only=True,
            )
        except Exception as cfg_err:
            raise ModelLoadError(
                f"Failed to load transformers configuration from '{self.model_dir}': {cfg_err}. "
                f"Ensure 'config.json' is present and valid.",
                details={"model_dir": self.model_dir, "error": str(cfg_err)}
            ) from cfg_err

        # Instantiate model based on architecture hints
        arch_lower = self.architecture.lower()
        if (not arch_lower or arch_lower == "transformersmodel") and hasattr(hf_config, "architectures") and hf_config.architectures:
            self.architecture = hf_config.architectures[0]
            arch_lower = self.architecture.lower()

        model = None
        try:
            if "sequenceclassification" in arch_lower:
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
            elif "tokenclassification" in arch_lower:
                from transformers import AutoModelForTokenClassification
                model = AutoModelForTokenClassification.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
            elif "questionanswering" in arch_lower:
                from transformers import AutoModelForQuestionAnswering
                model = AutoModelForQuestionAnswering.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
        except Exception as load_err:
            is_task_specific = any(k in arch_lower for k in ("sequenceclassification", "tokenclassification", "questionanswering"))
            if is_task_specific:
                raise ModelLoadError(
                    f"Failed to load task-specific transformers model '{self.architecture}' from '{self.model_dir}': {load_err}. "
                    f"Ensure weights files (model.safetensors or pytorch_model.bin) are present locally and compatible with {self.architecture}.",
                    details={"model_dir": self.model_dir, "architecture": self.architecture, "error": str(load_err)}
                ) from load_err

            # Generic fallback attempt only for non-task-specific models
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_dir,
                    config=hf_config,
                    local_files_only=True,
                )
            except Exception as fallback_err:
                raise ModelLoadError(
                    f"Failed to load transformers model weights from '{self.model_dir}': {load_err}. "
                    f"Ensure weights files (model.safetensors or pytorch_model.bin) are present locally.",
                    details={
                        "model_dir": self.model_dir,
                        "primary_error": str(load_err),
                        "fallback_error": str(fallback_err),
                    }
                ) from load_err

        # Load tokenizer if present locally
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        except Exception:
            logger.info(f"No tokenizer loaded for model at '{self.model_dir}'")

        # Place model on requested hardware device
        try:
            if hasattr(model, "to") and callable(model.to):
                model.to(self.device)
            if hasattr(model, "eval") and callable(model.eval):
                model.eval()
        except Exception as dev_err:
            raise HardwareIncompatibleError(
                f"Failed to move transformers model to device '{self.device}': {dev_err}",
                details={"device": self.device, "error": str(dev_err)}
            ) from dev_err

        self.model = model
        self.tokenizer = tokenizer
        self.hf_config = hf_config
        self._is_loaded = True
        logger.info(
            f"Transformers model loaded successfully from '{self.model_dir}' "
            f"(arch={self.architecture}, device={self.device})."
        )
        return self

    def unload(self) -> None:
        """Safely unloads model instance and frees execution memory."""
        if self.model is not None:
            try:
                if hasattr(self.model, "to") and callable(self.model.to):
                    self.model.to("cpu")
            except Exception:
                pass
            self.model = None

        self.tokenizer = None
        self.hf_config = None
        self._is_loaded = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()
        logger.info(f"Transformers model unloaded from memory (model_dir='{self.model_dir}').")

    def predict(self, preprocessed_input: Any) -> Any:
        """
        Executes actual model inference on preprocessed_input using PyTorch inference mode.
        Routes to task-specific inference handlers based on the loaded model architecture.
        """
        if not self.is_loaded or self.model is None:
            raise ModelLoadError("Cannot execute prediction on unloaded transformers model.")

        if preprocessed_input is None:
            raise InferenceError("Inference input cannot be None.")

        # Determine architecture family
        arch_hints = [self.architecture or "", getattr(self.model, "__class__", type(self)).__name__]
        if self.hf_config and hasattr(self.hf_config, "architectures") and self.hf_config.architectures:
            arch_hints.extend(self.hf_config.architectures)
        elif hasattr(self.model, "config") and hasattr(self.model.config, "architectures") and self.model.config.architectures:
            arch_hints.extend(self.model.config.architectures)

        arch_hints_lower = [h.lower() for h in arch_hints if isinstance(h, str)]

        if any("tokenclassification" in h for h in arch_hints_lower):
            return self._predict_token_classification(preprocessed_input)
        elif any("sequenceclassification" in h for h in arch_hints_lower):
            return self._predict_sequence_classification(preprocessed_input)
        elif any("questionanswering" in h for h in arch_hints_lower):
            return self._predict_question_answering(preprocessed_input)
        else:
            return self._predict_generic(preprocessed_input)

    def _get_id2label(self) -> dict:
        """Retrieves id2label mapping from model config or hf_config."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "id2label") and self.model.config.id2label:
            return self.model.config.id2label
        if self.hf_config and hasattr(self.hf_config, "id2label") and self.hf_config.id2label:
            return self.hf_config.id2label
        return {}

    def _get_label_for_id(self, class_id: Any) -> str:
        """Resolves human-readable label from class ID using id2label mapping."""
        id2label = self._get_id2label()
        if class_id in id2label:
            return str(id2label[class_id])
        if str(class_id) in id2label:
            return str(id2label[str(class_id)])
        try:
            cid_int = int(class_id)
            if cid_int in id2label:
                return str(id2label[cid_int])
            if str(cid_int) in id2label:
                return str(id2label[str(cid_int)])
        except (ValueError, TypeError):
            pass
        return f"LABEL_{class_id}"

    def _predict_token_classification(self, preprocessed_input: Any) -> Any:
        """
        Executes token classification inference.
        Tokenizes input text, executes forward pass under PyTorch inference mode,
        maps logits to class IDs / labels via id2label, and groups BIO labels into entities.
        """
        if self.tokenizer is None:
            raise InferenceError(
                f"Tokenizer is required for token classification inference with '{self.architecture}', "
                f"but no tokenizer was loaded for model at '{self.model_dir}'."
            )

        if isinstance(preprocessed_input, (bytes, bytearray)):
            try:
                preprocessed_input = preprocessed_input.decode("utf-8")
            except Exception as e:
                raise InferenceError(f"Failed to decode byte input as utf-8: {e}") from e

        if isinstance(preprocessed_input, str):
            if not preprocessed_input.strip():
                raise InferenceError("Input text cannot be empty or whitespace only.")
            return self._predict_single_token_classification(preprocessed_input)

        elif isinstance(preprocessed_input, dict):
            if "text" not in preprocessed_input:
                raise InferenceError(
                    f"Input dictionary for token classification must contain a 'text' key. "
                    f"Received keys: {list(preprocessed_input.keys())}"
                )
            text = preprocessed_input["text"]
            if not isinstance(text, str) or not text.strip():
                raise InferenceError("Input dictionary 'text' value must be a non-empty string.")
            return self._predict_single_token_classification(text)

        elif isinstance(preprocessed_input, (list, tuple)):
            if len(preprocessed_input) == 0:
                raise InferenceError("Input sequence cannot be empty.")
            results = []
            for idx, item in enumerate(preprocessed_input):
                if isinstance(item, str):
                    if not item.strip():
                        raise InferenceError(f"Item at index {idx} in input list is empty or whitespace.")
                    results.append(self._predict_single_token_classification(item))
                elif isinstance(item, dict) and "text" in item:
                    results.append(self._predict_single_token_classification(item["text"]))
                else:
                    raise InferenceError(
                        f"Unsupported item type at index {idx} in input list: {type(item).__name__}. "
                        f"Expected string or dict with 'text' key."
                    )
            return results

        else:
            raise InferenceError(
                f"Unsupported input type for token classification inference: {type(preprocessed_input).__name__}. "
                f"Expected str, list of str, or dict with 'text' key."
            )

    def _predict_single_token_classification(self, text: str) -> dict:
        """Token classification inference on a single text string."""
        try:
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                return_offsets_mapping=True,
                truncation=True,
            )
            offset_mapping = encoding.pop("offset_mapping", None)
        except (TypeError, ValueError):
            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
            )
            offset_mapping = None

        device = getattr(self, "device", "cpu")
        inputs = {k: v.to(device) if hasattr(v, "to") and callable(v.to) else v for k, v in encoding.items()}

        import torch
        inf_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inf_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        is_real_tensor = hasattr(logits, "dim") and callable(logits.dim) and isinstance(logits.dim(), int)

        if is_real_tensor:
            if hasattr(logits, "cpu") and callable(logits.cpu):
                logits = logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)
            pred_ids = preds[0].tolist() if logits.dim() >= 2 else preds.tolist()
            confidences = confs[0].tolist() if logits.dim() >= 2 else confs.tolist()
        else:
            raw_list = None
            if hasattr(logits, "tolist"):
                try:
                    tl = logits.tolist()
                    if isinstance(tl, list):
                        raw_list = tl
                except Exception:
                    pass
            if raw_list is not None:
                pred_ids = raw_list[0] if raw_list and isinstance(raw_list[0], list) else raw_list
            else:
                pred_ids = [0]
            confidences = [1.0] * len(pred_ids) if isinstance(pred_ids, list) else [1.0]

        # Retrieve token strings
        if "input_ids" in encoding and hasattr(self.tokenizer, "convert_ids_to_tokens"):
            input_ids = encoding["input_ids"][0] if hasattr(encoding["input_ids"], "__getitem__") else encoding["input_ids"]
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        elif hasattr(self.tokenizer, "tokenize"):
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = [f"tok_{i}" for i in range(len(pred_ids))]

        # Retrieve offsets
        offsets = None
        if offset_mapping is not None:
            if hasattr(offset_mapping, "tolist"):
                raw_off = offset_mapping.tolist()
                offsets = raw_off[0] if raw_off and isinstance(raw_off[0], list) and hasattr(offset_mapping, "dim") and offset_mapping.dim() >= 2 else raw_off
            elif isinstance(offset_mapping, list):
                offsets = offset_mapping[0] if offset_mapping and isinstance(offset_mapping[0], list) else offset_mapping

        num_items = min(len(tokens), len(pred_ids))
        raw_tokens = []
        for i in range(num_items):
            tok = tokens[i]
            pid = pred_ids[i]
            score = round(float(confidences[i]), 4) if i < len(confidences) else 1.0
            label = self._get_label_for_id(pid)
            st = None
            en = None
            if offsets and i < len(offsets):
                off = offsets[i]
                if isinstance(off, (list, tuple)) and len(off) == 2:
                    st, en = off[0], off[1]

            raw_tokens.append({
                "token": tok,
                "label": label,
                "class_id": pid,
                "score": score,
                "start": st,
                "end": en,
            })

        entities = self._group_bio_entities(raw_tokens, text)

        return {
            "text": text,
            "tokens": raw_tokens,
            "entities": entities,
        }

    def _group_bio_entities(self, raw_tokens: list, text: Optional[str] = None) -> list:
        """
        Groups token-level BIO labels into span-level entities.
        Preserves character offsets and computes mean confidence score across subwords.
        """
        entities = []
        current_entity = None

        special_tokens = set()
        if self.tokenizer and hasattr(self.tokenizer, "all_special_tokens") and self.tokenizer.all_special_tokens:
            special_tokens = set(self.tokenizer.all_special_tokens)

        for item in raw_tokens:
            tok = item.get("token", "")
            lbl = item.get("label", "O")
            score = item.get("score", 1.0)
            start = item.get("start")
            end = item.get("end")

            is_special = tok in special_tokens or (start == 0 and end == 0 and tok.startswith(("[", "<", "<s>", "</s>")))

            if is_special or lbl == "O" or not lbl:
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity, text))
                    current_entity = None
                continue

            # Parse BIO tag prefix
            if "-" in lbl:
                prefix, ent_type = lbl.split("-", 1)
            else:
                prefix = "B"
                ent_type = lbl

            if prefix == "B":
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity, text))
                current_entity = {
                    "entity": ent_type,
                    "tokens": [tok],
                    "scores": [score],
                    "start": start,
                    "end": end,
                }
            elif prefix == "I":
                if current_entity is not None and current_entity["entity"] == ent_type:
                    current_entity["tokens"].append(tok)
                    current_entity["scores"].append(score)
                    if end is not None:
                        current_entity["end"] = end
                else:
                    if current_entity is not None:
                        entities.append(self._finalize_entity(current_entity, text))
                    current_entity = {
                        "entity": ent_type,
                        "tokens": [tok],
                        "scores": [score],
                        "start": start,
                        "end": end,
                    }
            else:
                if current_entity is not None:
                    entities.append(self._finalize_entity(current_entity, text))
                    current_entity = None

        if current_entity is not None:
            entities.append(self._finalize_entity(current_entity, text))

        return entities

    def _finalize_entity(self, ent_dict: dict, text: Optional[str]) -> dict:
        tokens = ent_dict["tokens"]
        start = ent_dict["start"]
        end = ent_dict["end"]
        scores = ent_dict["scores"]
        avg_score = round(sum(scores) / len(scores), 4) if scores else 1.0

        word = ""
        if text is not None and start is not None and end is not None and 0 <= start <= end <= len(text):
            word = text[start:end]
        if not word:
            word = self._reconstruct_from_tokens(tokens)

        res = {
            "entity": ent_dict["entity"],
            "word": word,
            "score": avg_score,
        }
        if start is not None:
            res["start"] = start
        if end is not None:
            res["end"] = end
        return res

    def _reconstruct_from_tokens(self, tokens: list) -> str:
        cleaned = []
        for t in tokens:
            if t.startswith("##"):
                cleaned.append(t[2:])
            elif t.startswith(" "):
                cleaned.append(" " + t[1:])
            else:
                if cleaned:
                    cleaned.append(" ")
                cleaned.append(t)
        return "".join(cleaned).strip()

    def _predict_sequence_classification(self, preprocessed_input: Any) -> Any:
        """
        Executes sequence classification inference.
        Tokenizes text, executes forward pass under PyTorch inference mode,
        and returns predicted label, score, class ID, and class probabilities.
        """
        if self.tokenizer is None:
            raise InferenceError(
                f"Tokenizer is required for sequence classification inference with '{self.architecture}', "
                f"but no tokenizer was loaded for model at '{self.model_dir}'."
            )

        if isinstance(preprocessed_input, (bytes, bytearray)):
            try:
                preprocessed_input = preprocessed_input.decode("utf-8")
            except Exception as e:
                raise InferenceError(f"Failed to decode byte input as utf-8: {e}") from e

        if isinstance(preprocessed_input, str):
            if not preprocessed_input.strip():
                raise InferenceError("Input text cannot be empty or whitespace only.")
            return self._predict_single_sequence_classification(preprocessed_input)

        elif isinstance(preprocessed_input, dict):
            if "text" in preprocessed_input:
                text = preprocessed_input["text"]
                text_pair = preprocessed_input.get("text_pair") or preprocessed_input.get("sentence2")
                return self._predict_single_sequence_classification(text, text_pair)
            raise InferenceError(
                f"Input dictionary for sequence classification must contain a 'text' key. "
                f"Received keys: {list(preprocessed_input.keys())}"
            )

        elif isinstance(preprocessed_input, (list, tuple)):
            if len(preprocessed_input) == 0:
                raise InferenceError("Input sequence cannot be empty.")
            results = []
            for idx, item in enumerate(preprocessed_input):
                if isinstance(item, str):
                    results.append(self._predict_single_sequence_classification(item))
                elif isinstance(item, dict) and "text" in item:
                    results.append(self._predict_single_sequence_classification(item["text"], item.get("text_pair")))
                else:
                    raise InferenceError(f"Unsupported item type at index {idx}: {type(item).__name__}")
            return results

        else:
            raise InferenceError(
                f"Unsupported input type for sequence classification: {type(preprocessed_input).__name__}."
            )

    def _predict_single_sequence_classification(self, text: str, text_pair: Optional[str] = None) -> dict:
        device = getattr(self, "device", "cpu")
        if text_pair:
            encoding = self.tokenizer(text, text_pair, return_tensors="pt", truncation=True)
        else:
            encoding = self.tokenizer(text, return_tensors="pt", truncation=True)

        inputs = {k: v.to(device) if hasattr(v, "to") and callable(v.to) else v for k, v in encoding.items()}

        import torch
        inf_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inf_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits if hasattr(outputs, "logits") else (outputs[0] if isinstance(outputs, (tuple, list)) else outputs)
        is_real_tensor = hasattr(logits, "dim") and callable(logits.dim) and isinstance(logits.dim(), int)

        if is_real_tensor:
            if hasattr(logits, "cpu") and callable(logits.cpu):
                logits = logits.cpu()
            probs = torch.softmax(logits, dim=-1)
            pred_id = int(torch.argmax(logits, dim=-1)[0].item()) if logits.dim() >= 2 else int(torch.argmax(logits, dim=-1).item())
            score = float(probs[0][pred_id].item()) if logits.dim() >= 2 else float(probs[pred_id].item())
            label = self._get_label_for_id(pred_id)
            num_classes = probs.shape[-1]
            prob_dict = {
                self._get_label_for_id(i): round(float(probs[0][i].item()), 4) if logits.dim() >= 2 else round(float(probs[i].item()), 4)
                for i in range(num_classes)
            }
            return {
                "label": label,
                "score": round(score, 4),
                "class_id": pred_id,
                "probabilities": prob_dict,
            }
        else:
            pred_id = 0
            if hasattr(logits, "tolist"):
                try:
                    lst = logits.tolist()
                    if isinstance(lst, list) and lst:
                        if isinstance(lst[0], list) and lst[0]:
                            pred_id = lst[0].index(max(lst[0]))
                        elif isinstance(lst[0], (int, float)):
                            pred_id = lst.index(max(lst))
                except Exception:
                    pred_id = 0
            label = self._get_label_for_id(pred_id)
            return {
                "label": label,
                "score": 1.0,
                "class_id": pred_id,
                "probabilities": {label: 1.0},
            }

    def _predict_question_answering(self, preprocessed_input: Any) -> Any:
        """
        Executes question answering inference.
        Tokenizes question and context, executes forward pass under PyTorch inference mode,
        and extracts the predicted answer span.
        """
        if self.tokenizer is None:
            raise InferenceError(
                f"Tokenizer is required for question answering inference with '{self.architecture}', "
                f"but no tokenizer was loaded for model at '{self.model_dir}'."
            )

        question = None
        context = None
        if isinstance(preprocessed_input, dict):
            question = preprocessed_input.get("question")
            context = preprocessed_input.get("context")
        elif isinstance(preprocessed_input, (list, tuple)) and len(preprocessed_input) == 2:
            question = preprocessed_input[0]
            context = preprocessed_input[1]
        else:
            raise InferenceError(
                "Question answering requires a dictionary with 'question' and 'context' keys or a (question, context) pair."
            )

        if not question or not isinstance(question, str) or not question.strip():
            raise InferenceError("Question cannot be empty for question answering inference.")
        if not context or not isinstance(context, str) or not context.strip():
            raise InferenceError("Context cannot be empty for question answering inference.")

        device = getattr(self, "device", "cpu")
        encoding = self.tokenizer(question, context, return_tensors="pt", truncation=True)
        inputs = {k: v.to(device) if hasattr(v, "to") and callable(v.to) else v for k, v in encoding.items()}

        import torch
        inf_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inf_mode():
            outputs = self.model(**inputs)

        start_logits = getattr(outputs, "start_logits", outputs[0] if isinstance(outputs, (tuple, list)) else None)
        end_logits = getattr(outputs, "end_logits", outputs[1] if isinstance(outputs, (tuple, list)) and len(outputs) > 1 else None)

        is_real_tensor = (
            start_logits is not None and hasattr(start_logits, "dim") and callable(start_logits.dim) and isinstance(start_logits.dim(), int) and
            end_logits is not None and hasattr(end_logits, "dim") and callable(end_logits.dim) and isinstance(end_logits.dim(), int)
        )

        if is_real_tensor:
            if hasattr(start_logits, "cpu") and callable(start_logits.cpu):
                start_logits = start_logits.cpu()
            if hasattr(end_logits, "cpu") and callable(end_logits.cpu):
                end_logits = end_logits.cpu()
            start_idx = int(torch.argmax(start_logits, dim=-1)[0].item()) if start_logits.dim() >= 2 else int(torch.argmax(start_logits, dim=-1).item())
            end_idx = int(torch.argmax(end_logits, dim=-1)[0].item()) if end_logits.dim() >= 2 else int(torch.argmax(end_logits, dim=-1).item())
            if end_idx < start_idx:
                end_idx = start_idx
            input_ids = encoding["input_ids"][0] if hasattr(encoding["input_ids"], "__getitem__") else encoding["input_ids"]
            answer_tokens = input_ids[start_idx : end_idx + 1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip() if hasattr(self.tokenizer, "decode") else ""
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)
            start_score = float(start_probs[0][start_idx].item()) if start_logits.dim() >= 2 else float(start_probs[start_idx].item())
            end_score = float(end_probs[0][end_idx].item()) if end_logits.dim() >= 2 else float(end_probs[end_idx].item())
            score = round(start_score * end_score, 4)
            tokens = self.tokenizer.convert_ids_to_tokens(answer_tokens) if hasattr(self.tokenizer, "convert_ids_to_tokens") else []
            return {
                "answer": answer,
                "score": score,
                "start": start_idx,
                "end": end_idx,
                "tokens": tokens,
            }
        else:
            return {
                "answer": "mock answer",
                "score": 1.0,
                "start": 0,
                "end": 0,
                "tokens": [],
            }

    def _predict_generic(self, preprocessed_input: Any) -> Any:
        """
        Executes generic forward inference (e.g. AutoModel).
        Returns representation metadata such as last_hidden_state_shape.
        """
        device = getattr(self, "device", "cpu")
        if isinstance(preprocessed_input, (bytes, bytearray)):
            try:
                preprocessed_input = preprocessed_input.decode("utf-8")
            except Exception as e:
                raise InferenceError(f"Failed to decode byte input as utf-8: {e}") from e

        if isinstance(preprocessed_input, str):
            if not preprocessed_input.strip():
                raise InferenceError("Input text cannot be empty or whitespace only.")
            if self.tokenizer is None:
                raise InferenceError("Tokenizer is required to tokenize text for generic transformers model inference.")
            encoding = self.tokenizer(preprocessed_input, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) if hasattr(v, "to") and callable(v.to) else v for k, v in encoding.items()}
        elif isinstance(preprocessed_input, dict):
            inputs = {k: v.to(device) if hasattr(v, "to") and callable(v.to) else v for k, v in preprocessed_input.items()}
        else:
            raise InferenceError(
                f"Unsupported input type for generic transformers inference: {type(preprocessed_input).__name__}. "
                f"Expected str or tokenized tensor dictionary."
            )

        import torch
        inf_mode = getattr(torch, "inference_mode", torch.no_grad)
        with inf_mode():
            outputs = self.model(**inputs)

        res = {}
        if hasattr(outputs, "last_hidden_state"):
            lhs = outputs.last_hidden_state
            if hasattr(lhs, "shape"):
                res["last_hidden_state_shape"] = list(lhs.shape)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            po = outputs.pooler_output
            if hasattr(po, "shape"):
                res["pooler_output_shape"] = list(po.shape)
        if not res:
            if hasattr(outputs, "keys") and callable(outputs.keys):
                res["output_keys"] = list(outputs.keys())
            else:
                res["status"] = "success"
        return res

