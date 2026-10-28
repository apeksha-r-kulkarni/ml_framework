"""
Output Processor implementation for the ML Framework.

Provides centralized output structuring, task-specific result normalization,
confidence filtering, and output serialization across different model backends.
Implements BaseOutputProcessor.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from ..interfaces.base import BaseOutputProcessor
from ..results.execution_result import ExecutionContext
from ..errors.exceptions import OutputProcessingError

logger = logging.getLogger("ml_framework.output.processor")


class OutputProcessor(BaseOutputProcessor):
    """
    Standard output processor for the ML Framework.

    Transforms, normalizes, and validates raw model and inference backend outputs
    into documented domain structures, ensuring consistent output contracts
    across speech transcription, language detection, audio enhancement/denoising,
    and text processing tasks.
    """

    def __init__(
        self,
        task: str = "auto",
        strip_whitespace: bool = True,
        confidence_threshold: Optional[float] = None,
        custom_formatters: Optional[Dict[str, Any]] = None,
    ):
        self.task = (task or "auto").lower().strip()
        self.strip_whitespace = bool(strip_whitespace)
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None else None
        )
        self.custom_formatters = dict(custom_formatters or {})

    def process(
        self,
        raw_output: Any,
        context: Optional[ExecutionContext] = None,
    ) -> Any:
        """
        Transforms raw inference output into the framework's documented output structure.

        Args:
            raw_output: Output from the inference engine or model forward pass.
            context: Optional execution context containing correlation ID and stage timings.

        Returns:
            Normalized and structured output payload.

        Raises:
            OutputProcessingError: If the output is empty, malformed, or indicates backend failure.
        """
        if raw_output is None:
            raise OutputProcessingError(
                "Inference output cannot be None.",
                details={"correlation_id": context.correlation_id if context else None}
            )

        # 1. Handle error response payloads
        if isinstance(raw_output, dict):
            if "error" in raw_output and not raw_output.get("success", True):
                err_msg = str(raw_output["error"])
                raise OutputProcessingError(
                    f"Inference response reported backend error: {err_msg}",
                    details={
                        "error": err_msg,
                        "correlation_id": context.correlation_id if context else None
                    }
                )

        # 2. Custom formatter override if registered
        if self.task in self.custom_formatters and callable(self.custom_formatters[self.task]):
            try:
                return self.custom_formatters[self.task](raw_output, context)
            except Exception as fmt_err:
                raise OutputProcessingError(
                    f"Custom formatter failed for task '{self.task}': {fmt_err}",
                    details={"task": self.task, "error": str(fmt_err)}
                ) from fmt_err

        # 3. Task-specific normalization
        effective_task = self._resolve_task(raw_output)

        if effective_task == "transcription":
            return self._format_transcription(raw_output, context)
        elif effective_task == "language_detection":
            return self._format_language_detection(raw_output, context)
        elif effective_task == "denoising":
            return self._format_denoising(raw_output, context)

        # 4. Standard string normalization
        if isinstance(raw_output, str):
            return raw_output.strip() if self.strip_whitespace else raw_output

        # 5. Passthrough for structured types (lists, tuples, dicts, primitives)
        return raw_output

    def _resolve_task(self, raw_output: Any) -> str:
        """Determines the effective domain task from configuration or output structure."""
        if self.task != "auto":
            return self.task

        if isinstance(raw_output, dict):
            if "task" in raw_output:
                return str(raw_output["task"]).lower().strip()
            if any(k in raw_output for k in ("transcription", "transcript", "text")):
                return "transcription"
            if any(k in raw_output for k in ("language", "label", "lang")):
                return "language_detection"
            if any(k in raw_output for k in ("enhanced_audio", "clean_audio")):
                return "denoising"

        return "auto"

    def _format_transcription(
        self,
        raw_output: Any,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Normalizes transcription output into documented contract:
        {"task": "transcription", "text": str, "transcription": str, ...}
        """
        if isinstance(raw_output, dict):
            # Extract transcript text
            text = (
                raw_output.get("transcription")
                or raw_output.get("transcript")
                or raw_output.get("text")
                or ""
            )
            if isinstance(text, list):
                text = " ".join(str(item) for item in text)
            else:
                text = str(text)

            if self.strip_whitespace:
                text = text.strip()

            result = dict(raw_output)
            result["task"] = "transcription"
            result["transcription"] = text
            result["text"] = text
            return result

        elif isinstance(raw_output, str):
            text = raw_output.strip() if self.strip_whitespace else raw_output
            return {
                "task": "transcription",
                "text": text,
                "transcription": text,
            }

        elif isinstance(raw_output, (list, tuple)):
            text = " ".join(str(x) for x in raw_output)
            if self.strip_whitespace:
                text = text.strip()
            return {
                "task": "transcription",
                "text": text,
                "transcription": text,
            }

        raise OutputProcessingError(
            f"Cannot format output of type '{type(raw_output).__name__}' as transcription.",
            details={"raw_output_type": type(raw_output).__name__}
        )

    def _format_language_detection(
        self,
        raw_output: Any,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Normalizes language detection output into documented contract:
        {"task": "language_detection", "language": str, "label": str, "confidence": float, ...}
        """
        if isinstance(raw_output, dict):
            lang = str(
                raw_output.get("language")
                or raw_output.get("label")
                or raw_output.get("lang")
                or "unknown"
            ).strip()
            conf = float(raw_output.get("confidence", 1.0))

            if self.confidence_threshold is not None and conf < self.confidence_threshold:
                logger.info(
                    f"Language detection confidence {conf:.3f} below threshold {self.confidence_threshold:.3f}; marking unconfident."
                )

            result = dict(raw_output)
            result["task"] = "language_detection"
            result["language"] = lang
            result["label"] = lang
            result["confidence"] = conf
            return result

        elif isinstance(raw_output, str):
            lang = raw_output.strip()
            return {
                "task": "language_detection",
                "language": lang,
                "label": lang,
                "confidence": 1.0,
            }

        raise OutputProcessingError(
            f"Cannot format output of type '{type(raw_output).__name__}' as language detection.",
            details={"raw_output_type": type(raw_output).__name__}
        )

    def _format_denoising(
        self,
        raw_output: Any,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        Normalizes denoising/audio enhancement output into documented contract:
        {"task": "denoising", "enhanced_audio": np.ndarray, "sample_rate": int, ...}
        """
        if isinstance(raw_output, dict):
            audio = raw_output.get("enhanced_audio", raw_output.get("clean_audio"))
            if audio is None:
                raise OutputProcessingError(
                    "Denoising output missing 'enhanced_audio' or 'clean_audio' array.",
                    details={"keys": list(raw_output.keys())}
                )
            if np is not None and not isinstance(audio, np.ndarray):
                audio = np.asarray(audio, dtype=np.float32)

            sr = int(raw_output.get("sample_rate", 16000))
            result = dict(raw_output)
            result["task"] = "denoising"
            result["enhanced_audio"] = audio
            result["sample_rate"] = sr
            return result

        elif (np is not None and isinstance(raw_output, np.ndarray)) or isinstance(raw_output, list):
            audio = np.asarray(raw_output, dtype=np.float32) if np is not None else raw_output
            return {
                "task": "denoising",
                "enhanced_audio": audio,
                "sample_rate": 16000,
            }

        raise OutputProcessingError(
            f"Cannot format output of type '{type(raw_output).__name__}' as denoised audio.",
            details={"raw_output_type": type(raw_output).__name__}
        )

    @staticmethod
    def serialize(output: Any) -> Any:
        """
        Recursively converts complex data structures (NumPy ndarrays, scalar types)
        into JSON-serializable native Python primitives.
        """
        if np is not None:
            if isinstance(output, np.ndarray):
                return output.tolist()
            elif isinstance(output, (np.floating, np.float32, np.float64)):
                return float(output)
            elif isinstance(output, (np.integer, np.int32, np.int64)):
                return int(output)
            elif isinstance(output, np.bool_):
                return bool(output)
        if isinstance(output, dict):
            return {k: OutputProcessor.serialize(v) for k, v in output.items()}
        elif isinstance(output, (list, tuple)):
            return [OutputProcessor.serialize(x) for x in output]
        elif isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return output

    @classmethod
    def to_dict(cls, output: Any) -> Dict[str, Any]:
        """Converts output payload to a dictionary representation."""
        if isinstance(output, dict):
            return cls.serialize(output)
        return {"result": cls.serialize(output)}

    @classmethod
    def to_json(cls, output: Any) -> str:
        """Serializes output payload to a JSON string."""
        return json.dumps(cls.serialize(output))
