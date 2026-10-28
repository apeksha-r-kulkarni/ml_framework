"""
Exception hierarchy for the ML Framework.

Provides typed exceptions for all lifecycle stages and error-handling flows
as defined in the Error Handling Sequence.
"""


class MLFrameworkError(Exception):
    """Base exception for all errors originating within the ML Framework."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(MLFrameworkError):
    """Raised when configuration parameters are missing, invalid, or conflicting."""
    pass


class StateError(MLFrameworkError):
    """Raised when an operation is invoked on an invalid or unexpected lifecycle state."""
    pass


class ExecutionError(MLFrameworkError):
    """Raised when pipeline execution encounters an unrecoverable failure."""
    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when pipeline execution exceeds the configured timeout."""
    def __init__(self, message: str, timeout_seconds: float = None, details: dict = None):
        details = details or {}
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
        super().__init__(message, details=details)
        self.timeout_seconds = timeout_seconds


class EnvironmentError(MLFrameworkError):
    """Raised when hardware requirements or runtime dependencies are not satisfied."""
    pass


class DependencyMissingError(EnvironmentError):
    """Raised when required runtime packages are not installed in the offline environment."""
    def __init__(self, message: str, missing_packages: list = None, details: dict = None):
        details = details or {}
        if missing_packages:
            details["missing_packages"] = missing_packages
        super().__init__(message, details=details)
        self.missing_packages = missing_packages or []


class HardwareIncompatibleError(EnvironmentError):
    """Raised when the requested compute device (e.g. CUDA) is not available."""
    pass


class ModelLoadError(MLFrameworkError):
    """Raised when model artifact resolution, downloading, or framework instantiation fails."""
    pass


class ModelFormatError(ModelLoadError):
    """Raised when model artifact formats are unsupported, invalid, or corrupted."""
    pass


class StorageError(ModelLoadError):
    """Raised when remote or local artifact storage operations fail."""
    pass


class TransientStorageError(StorageError):
    """Raised for transient storage or network errors that may be resolved with backoff/retry."""
    pass


class InputProcessingError(MLFrameworkError):
    """Raised when input validation, sanitization, or modality preprocessing fails."""
    pass


class InvalidInputSourceError(InputProcessingError):
    """Raised when an input source path is missing, inaccessible, or has an unsupported scheme."""
    pass


class InferenceError(MLFrameworkError):
    """Raised when model forward pass or execution fails."""
    pass


class OutputProcessingError(MLFrameworkError):
    """Raised when raw model output transformation, formatting, or filtering fails."""
    pass


class ResourceCleanupError(MLFrameworkError):
    """Raised when resource deallocation, memory clearance, or cache cleanup fails."""
    pass
