"""
ML Framework - Standalone, reusable, offline-capable ML execution framework.

Designed for model loading, dynamic inference, and lifecycle management across
different application domains (Text Scraping, ASR, LID, etc.).
"""

__version__ = "0.1.0"

from . import ml
from .execution_manager import ExecutionManager
from .config.model_config import FrameworkConfig, StorageConfig, InputConfig, TritonConfig
from .results.execution_result import ExecutionResult, ExecutionContext, ExecutionState
from .input.processor import InputProcessor
from .output.processor import OutputProcessor
from .status.tracker import StatusTracker
from .environment.environment_manager import EnvironmentManager
from .models.model_loader import ModelLoader
from .inference.triton_engine import TritonInferenceEngine
from .inference.triton_client import TritonClient
from .inference.tensor_mapper import TensorMapper
from .models.specialized_loaders import (
    SpeechModelLoader,
    SpacyModelLoader,
    TransformersModelLoader,
    ONNXModelLoader,
)
from .models.speech_adapter import SpeechModelAdapter
from .models.spacy_adapter import SpacyModelAdapter
from .models.transformers_adapter import TransformersModelAdapter
from .models.onnx_adapter import ONNXModelAdapter
from .errors.exceptions import (
    MLFrameworkError,
    ConfigurationError,
    StateError,
    ExecutionError,
    ExecutionTimeoutError,
    EnvironmentError,
    DependencyMissingError,
    HardwareIncompatibleError,
    ModelLoadError,
    ModelFormatError,
    StorageError,
    TransientStorageError,
    InputProcessingError,
    InvalidInputSourceError,
    InferenceError,
    OutputProcessingError,
    ResourceCleanupError,
)

__all__ = [
    "__version__",
    "ml",
    "ExecutionManager",
    "FrameworkConfig",
    "StorageConfig",
    "InputConfig",
    "TritonConfig",
    "TritonInferenceEngine",
    "TritonClient",
    "TensorMapper",
    "ExecutionResult",
    "ExecutionContext",
    "ExecutionState",
    "InputProcessor",
    "OutputProcessor",
    "StatusTracker",
    "EnvironmentManager",
    "ModelLoader",
    "SpeechModelLoader",
    "SpacyModelLoader",
    "TransformersModelLoader",
    "ONNXModelLoader",
    "SpeechModelAdapter",
    "SpacyModelAdapter",
    "TransformersModelAdapter",
    "ONNXModelAdapter",
    "MLFrameworkError",
    "ConfigurationError",
    "StateError",
    "ExecutionError",
    "ExecutionTimeoutError",
    "EnvironmentError",
    "DependencyMissingError",
    "HardwareIncompatibleError",
    "ModelLoadError",
    "ModelFormatError",
    "StorageError",
    "TransientStorageError",
    "InputProcessingError",
    "InvalidInputSourceError",
    "InferenceError",
    "OutputProcessingError",
    "ResourceCleanupError",
]
