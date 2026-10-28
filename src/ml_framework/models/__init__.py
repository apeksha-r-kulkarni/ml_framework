"""
Model loading and adapter subsystem for the ML Framework.

Provides generic model resolution, format detection, framework adapters,
and specialized loaders for Speech, spaCy, Transformers, and ONNX models.
"""

from .format_detector import FormatDetector, SPEECH_MODEL_TYPES
from .speech_adapter import SpeechModelAdapter
from .spacy_adapter import SpacyModelAdapter
from .transformers_adapter import TransformersModelAdapter
from .onnx_adapter import ONNXModelAdapter
from .specialized_loaders import (
    BaseSpecializedModelLoader,
    SpeechModelLoader,
    SpacyModelLoader,
    TransformersModelLoader,
    ONNXModelLoader,
)
from .model_loader import ModelLoader

__all__ = [
    "FormatDetector",
    "SPEECH_MODEL_TYPES",
    "SpeechModelAdapter",
    "SpacyModelAdapter",
    "TransformersModelAdapter",
    "ONNXModelAdapter",
    "BaseSpecializedModelLoader",
    "SpeechModelLoader",
    "SpacyModelLoader",
    "TransformersModelLoader",
    "ONNXModelLoader",
    "ModelLoader",
]
