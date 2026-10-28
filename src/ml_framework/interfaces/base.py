"""
Component interface placeholders for the ML Framework.

Defines the abstract contracts for all lifecycle components to be implemented
across the eight assigned development tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from ..config.model_config import FrameworkConfig
from ..results.execution_result import ExecutionContext, ExecutionResult


class BaseEnvironmentManager(ABC):
    """Task 3 Interface: Hardware validation, device placement, and dependency checking."""

    @abstractmethod
    def validate_environment(self, config: FrameworkConfig) -> bool:
        """Validates that hardware and library dependencies meet configuration requirements."""
        pass

    @abstractmethod
    def get_device(self) -> str:
        """Returns the resolved hardware execution device ('cpu', 'cuda', etc.)."""
        pass

    def inspect_dependencies(self, model_source_or_metadata: Any, framework: str = "auto") -> list:
        """Inspects model metadata or configuration to determine required runtime dependencies."""
        return []


class BaseInputReader(ABC):
    """Abstract interface for source-specific input readers (local file, object storage, Kafka, direct)."""

    @abstractmethod
    def validate(self, source: Any) -> bool:
        """Validates that the source path or configuration is accessible and valid."""
        pass

    @abstractmethod
    def read(self, source: Any, batch_size: int = 1, chunk_size: Optional[int] = None):
        """Reads input data from the source, yielding items or chunks."""
        pass

    def close(self) -> None:
        """Releases underlying resources, file handles, or network connectors."""
        pass


class BaseModelLoader(ABC):
    """Task 4 Interface: Generic model loader and framework adapter resolver."""

    @abstractmethod
    def load(self, model_source: str, config: Optional[FrameworkConfig] = None) -> Any:
        """Resolves, downloads, detects, and instantiates the appropriate model adapter."""
        pass

    @abstractmethod
    def unload(self, model_instance: Any) -> None:
        """Safely unloads model instances and frees execution memory."""
        pass


class BaseModelAdapter(ABC):
    """Task 4 Interface: Common contract for all specialized framework adapters."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and active in memory."""
        pass

    @abstractmethod
    def load(self) -> "BaseModelAdapter":
        """Loads model weights and artifacts from local files only."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Safely unloads model instances and frees execution memory."""
        pass

    @abstractmethod
    def predict(self, preprocessed_input: Any) -> Any:
        """Executes or delegates forward prediction on the model instance."""
        pass


class BaseInputProcessor(ABC):
    """Task 2 Interface: Input validation, sanitization, and modality-specific preprocessing."""

    @abstractmethod
    def process(self, raw_input: Any, context: ExecutionContext) -> Any:
        """Prepares raw input (text, audio, tensors) for inference consumption."""
        pass

    def validate_source(self, source: Any) -> bool:
        """Validates input source accessibility before expensive operations. Returns True or raises InputProcessingError."""
        return True

    def close(self) -> None:
        """Releases underlying connectors, consumers, or reader resources."""
        pass


class BaseInferenceEngine(ABC):
    """Task 5 Interface: Model forward pass execution and batching core."""

    @abstractmethod
    def infer(self, model: Any, preprocessed_input: Any, context: ExecutionContext) -> Any:
        """Executes inference on the model instance with preprocessed input."""
        pass


class BaseOutputProcessor(ABC):
    """Task 6 Interface: Output structuring, confidence filtering, and result formatting."""

    @abstractmethod
    def process(self, raw_output: Any, context: ExecutionContext) -> Any:
        """Transforms raw model output into standardized, typed domain objects."""
        pass


class BaseStatusTracker(ABC):
    """Task 7 Interface: Runtime telemetry, state tracking, and status notification callbacks."""

    @abstractmethod
    def record_success(self, result: ExecutionResult) -> None:
        """Records execution success metrics and notifies registered listeners."""
        pass

    @abstractmethod
    def record_failure(self, error: Exception, context: ExecutionContext) -> None:
        """Records execution failure metrics and notifies registered listeners."""
        pass


class BaseResourceManager(ABC):
    """Task 8 Interface: Memory deallocation, GPU cache clearance, and artifact directory cleanup."""

    @abstractmethod
    def cleanup(self) -> None:
        """Executes garbage collection and releases cached device resources."""
        pass
