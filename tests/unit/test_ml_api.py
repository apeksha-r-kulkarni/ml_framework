"""
Unit tests for the required public ML facade API:
    from ml_framework import ml
    model = ml.load(...)
    model.process(...)
    model.unload()
    model.stop()
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import ml, FrameworkConfig, ExecutionState
from ml_framework.errors import (
    StateError,
    InvalidInputSourceError,
    InferenceError,
)
from ml_framework.interfaces import (
    BaseModelLoader,
    BaseInferenceEngine,
    BaseResourceManager,
)


class MockLoader(BaseModelLoader):
    def __init__(self):
        self.loaded = False
        self.unloaded = False

    def load(self, model_source: str, config=None):
        self.loaded = True
        return {"source": model_source}

    def unload(self, model_instance):
        self.unloaded = True


class MockInference(BaseInferenceEngine):
    def __init__(self, should_fail: bool = False):
        self.should_fail = should_fail

    def infer(self, model, preprocessed_input, context):
        if self.should_fail:
            raise InferenceError("Model execution failed internally")
        return f"PREDICTED_{preprocessed_input}"


class TestMLFacadeAPI(unittest.TestCase):
    """Test suite for ml.load, process, unload, and stop lifecycle operations."""

    def test_standard_ml_workflow(self):
        """Test the standard user journey: ml.load -> process -> unload -> stop."""
        loader = MockLoader()
        engine = MockInference()

        # 1. Load model via ml facade
        model = ml.load(
            model_source="local://dummy/model",
            model_loader=loader,
            inference_engine=engine,
        )

        self.assertTrue(model.is_ready)
        self.assertTrue(loader.loaded)
        self.assertEqual(model.state, ExecutionState.READY)

        # 2. Process input
        result = model.process("test input text")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "PREDICTED_test input text")
        self.assertGreaterEqual(result.duration_ms, 0)

        # Verify status output and data output are strictly distinguishable
        result_dict = result.to_dict()
        self.assertIn("output", result_dict)
        self.assertIn("state", result_dict)
        self.assertIn("duration_ms", result_dict)
        self.assertEqual(result_dict["state"], "READY")

        # 3. Unload model
        model.unload()
        self.assertTrue(loader.unloaded)
        self.assertEqual(model.state, ExecutionState.UNINITIALIZED)
        self.assertFalse(model.is_ready)

        # 4. Stop manager
        model.stop()
        self.assertEqual(model.state, ExecutionState.STOPPED)

        # Execution after stop must raise StateError
        with self.assertRaises(StateError):
            model.process("cannot execute after stop")

    def test_error_handling_never_marks_failed_execution_as_success(self):
        """
        Verify that failures in pipeline stages are routed through common error handling
        and NEVER marked as successful.
        """
        failing_engine = MockInference(should_fail=True)
        model = ml.load(inference_engine=failing_engine)

        result = model.process("bad data payload")

        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error_type, "InferenceError")
        self.assertIn("Model execution failed internally", result.error)
        self.assertEqual(result.state, ExecutionState.FAILED)

    def test_ml_load_pre_validation_rejects_invalid_input(self):
        """
        Verify that ml.load with an invalid input source rejects BEFORE model loading
        and cleans up gracefully.
        """
        loader = MockLoader()
        with self.assertRaises(InvalidInputSourceError):
            ml.load(
                model_source="local://dummy",
                input_source="/non/existent/path/for/pipeline.txt",
                model_loader=loader,
            )

        # Model loading should NEVER have been started
        self.assertFalse(loader.loaded)

    def test_helper_functions(self):
        """Test validate_input and validate_environment helpers."""
        self.assertTrue(ml.validate_input("Direct valid string"))
        with self.assertRaises(InvalidInputSourceError):
            ml.validate_input("/non/existent/file.csv")

        self.assertTrue(ml.validate_environment(FrameworkConfig(framework="auto")))


if __name__ == "__main__":
    unittest.main()
