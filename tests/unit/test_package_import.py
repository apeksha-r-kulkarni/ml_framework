"""
Unit tests verifying package importability, versioning, and public exports.
"""

import sys
import unittest
from pathlib import Path

# Ensure src/ is on python path for test execution without prior pip install
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


class TestPackageImport(unittest.TestCase):
    """Verifies that ml_framework imports cleanly with standard public API symbols."""

    def test_package_import(self):
        """Test that ml_framework can be imported without errors or side-effects."""
        import ml_framework
        self.assertIsNotNone(ml_framework)

    def test_version_attribute(self):
        """Test that __version__ is defined and conforms to semantic versioning."""
        import ml_framework
        self.assertTrue(hasattr(ml_framework, "__version__"))
        self.assertIsInstance(ml_framework.__version__, str)
        parts = ml_framework.__version__.split(".")
        self.assertGreaterEqual(len(parts), 2)

    def test_public_api_exports(self):
        """Test that required core classes and error types are exported directly."""
        from ml_framework import (
            ml,
            ExecutionManager,
            FrameworkConfig,
            StorageConfig,
            InputConfig,
            ExecutionResult,
            ExecutionContext,
            ExecutionState,
            InputProcessor,
            EnvironmentManager,
            ModelLoader,
            MLFrameworkError,
            ConfigurationError,
            StateError,
            ExecutionError,
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

        self.assertTrue(hasattr(ml, "load"))
        self.assertTrue(callable(ExecutionManager))
        self.assertTrue(callable(FrameworkConfig))
        self.assertTrue(callable(StorageConfig))
        self.assertTrue(callable(InputConfig))
        self.assertTrue(callable(ExecutionResult))
        self.assertTrue(callable(ExecutionContext))
        self.assertTrue(callable(InputProcessor))
        self.assertTrue(callable(EnvironmentManager))
        self.assertTrue(callable(ModelLoader))
        self.assertTrue(issubclass(ExecutionState, object))
        self.assertTrue(issubclass(MLFrameworkError, Exception))
        self.assertTrue(issubclass(ConfigurationError, MLFrameworkError))
        self.assertTrue(issubclass(InferenceError, MLFrameworkError))
        self.assertTrue(issubclass(InvalidInputSourceError, InputProcessingError))
        self.assertTrue(issubclass(DependencyMissingError, EnvironmentError))
        self.assertTrue(issubclass(HardwareIncompatibleError, EnvironmentError))

    def test_all_declaration(self):
        """Verify __all__ contains only accessible symbols."""
        import ml_framework
        self.assertTrue(hasattr(ml_framework, "__all__"))
        for symbol in ml_framework.__all__:
            self.assertTrue(hasattr(ml_framework, symbol), f"Symbol '{symbol}' in __all__ not found in package")


if __name__ == "__main__":
    unittest.main()
