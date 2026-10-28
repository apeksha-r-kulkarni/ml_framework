"""
Mocked Integration Tests for External Pipeline Integration Readiness.

Simulates an independent consumer (e.g., Text Pipeline, ASR Worker, LID Client)
interacting strictly through the public API of ml_framework:
- ml.load(), ml.validate_input(), ml.validate_environment()
- model.process(), model.unload(), model.stop()
- ExecutionManager(config=...).initialize(), execute(), shutdown()
- FrameworkConfig, StorageConfig, InputConfig
- ExecutionContext, ExecutionResult, ExecutionState

Note: Uses controlled test doubles (mocks/stubs) for model loading, inference execution,
and resource management to verify external integration contracts, error containment,
timeout semantics, and lifecycle teardown without requiring real model weights or
heavy third-party ML runtime installations.
"""

import os
import sys
import time
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional, Dict, List

# Ensure src/ is on python path for test execution
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import (
    ml,
    ExecutionManager,
    EnvironmentManager,
    FrameworkConfig,
    InputConfig,
    StorageConfig,
    ExecutionContext,
    ExecutionResult,
    ExecutionState,
)
from ml_framework.interfaces import (
    BaseModelLoader,
    BaseInferenceEngine,
    BaseOutputProcessor,
    BaseResourceManager,
)
from ml_framework.errors import (
    ConfigurationError,
    StateError,
    InvalidInputSourceError,
    InferenceError,
    ModelLoadError,
    DependencyMissingError,
    ExecutionTimeoutError,
    MLFrameworkError,
)


# -----------------------------------------------------------------------------
# Controlled Test Doubles for Pipeline Consumer Simulation
# -----------------------------------------------------------------------------

class PipelineTestModelLoader(BaseModelLoader):
    """
    Test double simulating model weight loading and artifact resolution.
    Explicitly labeled: does not perform real model deserialization.
    """
    def __init__(self, should_fail: bool = False, failure_message: str = "Simulated artifact error"):
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.loaded_sources: List[str] = []
        self.unloaded_instances: List[Any] = []
        self.load_count = 0
        self.unload_count = 0

    def load(self, model_source: str, config: Optional[FrameworkConfig] = None) -> Any:
        self.load_count += 1
        if self.should_fail:
            raise ModelLoadError(self.failure_message)
        self.loaded_sources.append(model_source)
        # Return a synthetic model instance runner representation
        return {
            "model_type": "text_ner_pipeline_mock",
            "source": model_source,
            "vocab_size": 50000,
        }

    def unload(self, model_instance: Any) -> None:
        self.unload_count += 1
        self.unloaded_instances.append(model_instance)


class PipelineTestInferenceEngine(BaseInferenceEngine):
    """
    Test double simulating model forward pass inference.
    Explicitly labeled: generates synthetic predictions for contract testing.
    """
    def __init__(
        self,
        should_fail: bool = False,
        failure_message: str = "Inference forward pass failed",
        delay_seconds: float = 0.0,
    ):
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.delay_seconds = delay_seconds
        self.infer_calls: List[Any] = []

    def infer(self, model: Any, preprocessed_input: Any, context: ExecutionContext) -> Any:
        self.infer_calls.append(preprocessed_input)
        if self.delay_seconds > 0.0:
            time.sleep(self.delay_seconds)
        if self.should_fail:
            raise InferenceError(self.failure_message)

        # Produce realistic structured prediction contract
        if isinstance(preprocessed_input, str):
            words = preprocessed_input.split()
            entities = []
            for word in words:
                if word and word[0].isupper():
                    entities.append({"token": word, "label": "PROPN", "score": 0.98})
            return {"text": preprocessed_input, "entities": entities, "token_count": len(words)}
        elif isinstance(preprocessed_input, list):
            return [{"item": item, "processed": True} for item in preprocessed_input]
        return {"raw_input": preprocessed_input, "status": "predicted"}


class PipelineTestResourceManager(BaseResourceManager):
    """
    Test double simulating accelerator memory cleanup and temporary directory purges.
    """
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


# -----------------------------------------------------------------------------
# Integration Test Cases
# -----------------------------------------------------------------------------

class TestPipelineInitialization(unittest.TestCase):
    """
    1. Tests initialization using valid configurations via public API entry points.
    """

    def test_initialization_via_ml_facade_standard(self):
        """Simulate an external pipeline initializing the framework via ml.load() with default environment."""
        loader = PipelineTestModelLoader()
        engine = PipelineTestInferenceEngine()

        model = ml.load(
            model_source="local://models/ner_v1",
            framework="auto",
            device="cpu",
            batch_size=2,
            timeout_seconds=15.0,
            model_loader=loader,
            inference_engine=engine,
        )

        self.assertIsInstance(model, ExecutionManager)
        self.assertTrue(model.is_ready)
        self.assertEqual(model.state, ExecutionState.READY)
        self.assertEqual(model.model_id, "local://models/ner_v1")
        self.assertEqual(loader.load_count, 1)
        self.assertEqual(loader.loaded_sources[0], "local://models/ner_v1")

        # Clean teardown
        model.stop()

    def test_initialization_with_framework_and_mocked_env(self):
        """Simulate an external pipeline specifying heavy framework (transformers) with mocked environment."""
        loader = PipelineTestModelLoader()
        engine = PipelineTestInferenceEngine()
        env_mgr = EnvironmentManager(override_packages={"transformers": True, "torch": True})

        model = ml.load(
            model_source="local://models/ner_v1",
            framework="transformers",
            device="cpu",
            batch_size=2,
            timeout_seconds=15.0,
            env_manager=env_mgr,
            model_loader=loader,
            inference_engine=engine,
        )

        self.assertTrue(model.is_ready)
        self.assertEqual(model.state, ExecutionState.READY)
        model.stop()

    def test_initialization_via_framework_config_object(self):
        """Simulate an external pipeline supplying a typed FrameworkConfig object with object storage."""
        loader = PipelineTestModelLoader()
        env_mgr = EnvironmentManager(override_packages={"boto3": True})
        cfg = FrameworkConfig(
            model_name="text_ner",
            model_version="2.0.1",
            model_source="minio://models/ner_model.tar.gz",
            device="cpu",
            batch_size=4,
            timeout_seconds=20.0,
        )

        model = ml.load(config=cfg, env_manager=env_mgr, model_loader=loader)

        self.assertTrue(model.is_ready)
        self.assertEqual(model.model_id, "text_ner:2.0.1")
        self.assertEqual(model.config.batch_size, 4)
        self.assertEqual(model.config.timeout_seconds, 20.0)

        model.stop()

    def test_initialization_via_dict_config(self):
        """Simulate an external pipeline supplying configuration as a dictionary."""
        loader = PipelineTestModelLoader()
        config_dict = {
            "model_name": "lid_classifier",
            "model_version": "1.0",
            "model_source": "file:///opt/models/lid",
            "device": "cpu",
            "timeout_seconds": 10.0,
        }

        model = ml.load(config=config_dict, model_loader=loader)

        self.assertTrue(model.is_ready)
        self.assertEqual(model.model_id, "lid_classifier:1.0")
        model.stop()


class TestPipelineValidInputProcessing(unittest.TestCase):
    """
    2. Tests processing valid inputs with controlled test doubles for model loading and inference.
    """

    def setUp(self):
        self.loader = PipelineTestModelLoader()
        self.engine = PipelineTestInferenceEngine()
        self.model = ml.load(
            model_source="local://models/ner",
            model_loader=self.loader,
            inference_engine=self.engine,
        )

    def tearDown(self):
        if self.model.state != ExecutionState.STOPPED:
            self.model.stop()

    def test_process_single_text_payload(self):
        """Consumer submits a raw text string for entity extraction."""
        text_input = "Apple acquired a London startup in Europe"
        result = self.model.process(text_input)

        self.assertIsInstance(result, ExecutionResult)
        self.assertTrue(result.success)
        self.assertEqual(result.state, ExecutionState.READY)
        self.assertIsNone(result.error)
        self.assertIsNone(result.error_type)

        # Check inference output structure
        self.assertIsInstance(result.output, dict)
        self.assertEqual(result.output["text"], text_input)
        self.assertEqual(len(result.output["entities"]), 3)
        tokens = [ent["token"] for ent in result.output["entities"]]
        self.assertIn("Apple", tokens)
        self.assertIn("London", tokens)
        self.assertIn("Europe", tokens)

    def test_process_collection_payload(self):
        """Consumer submits a list of text records for batch execution."""
        batch_input = ["First document text", "Second document text"]
        result = self.model.process(batch_input)

        self.assertTrue(result.success)
        self.assertIsInstance(result.output, list)
        self.assertEqual(len(result.output), 2)
        self.assertEqual(result.output[0]["item"], "First document text")

    def test_process_local_file_source(self):
        """Consumer provides a valid local file path as input."""
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as tmp:
            tmp.write("Content line from input file\n")
            tmp_path = tmp.name

        try:
            result = self.model.process(tmp_path)
            self.assertTrue(result.success)
            self.assertEqual(result.output["text"], "Content line from input file")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPipelineResultMetadataAndContext(unittest.TestCase):
    """
    3. Tests access to output and result metadata (correlation IDs, stage durations, to_dict).
    """

    def setUp(self):
        self.loader = PipelineTestModelLoader()
        self.engine = PipelineTestInferenceEngine()
        self.model = ml.load(
            model_source="local://models/ner_v1",
            model_loader=self.loader,
            inference_engine=self.engine,
        )

    def tearDown(self):
        self.model.stop()

    def test_correlation_id_and_custom_metadata_propagation(self):
        """Consumer supplies an ExecutionContext with custom correlation ID and metadata."""
        custom_corr_id = "text-pipeline-tx-2026-9901"
        consumer_metadata = {
            "source_topic": "news_feed",
            "document_id": "doc_8831",
            "priority": "high",
        }
        ctx = ExecutionContext(
            correlation_id=custom_corr_id,
            metadata=consumer_metadata,
        )

        result = self.model.process("Sample document text", context=ctx)

        self.assertTrue(result.success)
        # Verify correlation ID is preserved
        self.assertEqual(result.context.correlation_id, custom_corr_id)
        # Verify custom metadata is preserved
        self.assertEqual(result.context.metadata["source_topic"], "news_feed")
        self.assertEqual(result.context.metadata["document_id"], "doc_8831")

    def test_stage_durations_tracking(self):
        """Verify per-stage latency telemetry is populated in milliseconds."""
        result = self.model.process("Sample text payload")

        self.assertTrue(result.success)
        self.assertGreaterEqual(result.duration_ms, 0.0)

        # Stage durations must record all three core pipeline stages
        durations = result.context.stage_durations
        self.assertIn("input_processing", durations)
        self.assertIn("inference", durations)
        self.assertIn("output_processing", durations)

        for stage, duration in durations.items():
            self.assertIsInstance(duration, float)
            self.assertGreaterEqual(duration, 0.0)

    def test_result_to_dict_serialization(self):
        """Verify ExecutionResult.to_dict() adheres to the documented schema."""
        ctx = ExecutionContext(correlation_id="corr-json-test")
        result = self.model.process("JSON test input", context=ctx)

        res_dict = result.to_dict()
        self.assertIsInstance(res_dict, dict)
        self.assertEqual(res_dict["success"], True)
        self.assertEqual(res_dict["state"], "READY")
        self.assertIn("output", res_dict)
        self.assertIn("duration_ms", res_dict)
        self.assertIn("applied_model_id", res_dict)
        self.assertEqual(res_dict["context"]["correlation_id"], "corr-json-test")
        self.assertIn("stage_durations", res_dict["context"])


class TestPipelineInvalidConfigurationAndInput(unittest.TestCase):
    """
    4. Tests invalid configurations and inputs are rejected cleanly.
    """

    def test_invalid_config_parameters_rejected(self):
        """Invalid batch size, timeout, or device raises ConfigurationError on load."""
        loader = PipelineTestModelLoader()

        # 1. Invalid batch_size
        with self.assertRaises(ConfigurationError) as cm:
            ml.load(batch_size=0, model_loader=loader)
        self.assertIn("batch_size must be >= 1", str(cm.exception))

        # 2. Invalid timeout_seconds
        with self.assertRaises(ConfigurationError) as cm:
            ml.load(timeout_seconds=-1.0, model_loader=loader)
        self.assertIn("timeout_seconds must be > 0", str(cm.exception))

        # 3. Invalid device
        with self.assertRaises(ConfigurationError) as cm:
            ml.load(device="invalid_quantum_unit", model_loader=loader)
        self.assertIn("Invalid device 'invalid_quantum_unit'", str(cm.exception))

    def test_invalid_input_source_fails_fast_before_model_loading(self):
        """
        Supplying an invalid input_source to ml.load() fails fast:
        raises InvalidInputSourceError and NEVER triggers expensive model loading.
        """
        loader = PipelineTestModelLoader()
        non_existent_path = "/non/existent/path/for/pipeline_input.jsonl"

        with self.assertRaises(InvalidInputSourceError):
            ml.load(
                model_source="local://models/expensive_model",
                input_source=non_existent_path,
                model_loader=loader,
            )

        # Model loading must never have been attempted
        self.assertEqual(loader.load_count, 0)

    def test_empty_string_input_rejected_at_runtime(self):
        """Passing empty or whitespace-only string returns non-crash failure result."""
        model = ml.load(model_source="local://models/dummy")
        result = model.process("   ")

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "InvalidInputSourceError")
        self.assertIn("cannot be empty or whitespace only", result.error)
        self.assertEqual(result.state, ExecutionState.FAILED)
        model.stop()

    def test_none_input_rejected_at_runtime(self):
        """Passing None input returns non-crash failure result."""
        model = ml.load(model_source="local://models/dummy")
        result = model.process(None)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "InvalidInputSourceError")
        self.assertIn("cannot be None", result.error)
        self.assertEqual(result.state, ExecutionState.FAILED)
        model.stop()

    def test_validate_input_helper_function(self):
        """Consumer can proactively validate input sources using ml.validate_input()."""
        self.assertTrue(ml.validate_input("Valid direct string"))
        with self.assertRaises(InvalidInputSourceError):
            ml.validate_input("/non/existent/input/file.txt")


class TestPipelineFailureHandling(unittest.TestCase):
    """
    5. Tests model-loading and inference failures are trapped without host crashes.
    """

    def test_model_loading_failure_during_initialization(self):
        """ModelLoader failure transitions manager to FAILED and raises ModelLoadError."""
        failing_loader = PipelineTestModelLoader(
            should_fail=True,
            failure_message="Checkpoint file is corrupted (CRC mismatch)"
        )

        with self.assertRaises(ModelLoadError) as cm:
            ml.load(
                model_source="local://models/broken_model",
                model_loader=failing_loader,
            )

        self.assertIn("Checkpoint file is corrupted", str(cm.exception))

    def test_missing_dependency_raises_clear_error(self):
        """Consumer requesting missing framework receives DependencyMissingError with offline install command."""
        env_mgr = EnvironmentManager(override_packages={"spacy": False})
        with self.assertRaises(DependencyMissingError) as cm:
            ml.load(
                framework="spacy",
                env_manager=env_mgr,
            )
        self.assertIn("spacy", str(cm.exception))
        self.assertIn("pip install --no-index --find-links=/path/to/offline/wheels", str(cm.exception))

    def test_inference_stage_failure_contained_in_result(self):
        """
        Failure during inference forward pass does NOT crash the consumer process;
        it returns ExecutionResult(success=False) with error details.
        """
        loader = PipelineTestModelLoader()
        failing_engine = PipelineTestInferenceEngine(
            should_fail=True,
            failure_message="CUDA out of memory error during forward pass"
        )

        model = ml.load(
            model_source="local://models/ner",
            model_loader=loader,
            inference_engine=failing_engine,
        )

        result = model.process("Valid text to be inferred")

        self.assertFalse(result.success)
        self.assertIsNone(result.output)
        self.assertEqual(result.error_type, "InferenceError")
        self.assertIn("CUDA out of memory", result.error)
        self.assertEqual(result.state, ExecutionState.FAILED)

        # Consumer remains able to call model on subsequent inputs without deadlock
        failing_engine.should_fail = False
        retry_result = model.process("Retry payload")
        self.assertTrue(retry_result.success)
        self.assertEqual(retry_result.state, ExecutionState.READY)

        model.stop()


class TestPipelineTimeoutBehavior(unittest.TestCase):
    """
    6. Tests timeout enforcement on slow pipeline stage executions.
    """

    def test_execution_timeout_returns_timeout_result(self):
        """
        When execution exceeds timeout_seconds, returns ExecutionResult with
        ExecutionTimeoutError without throwing unhandled exceptions.
        """
        loader = PipelineTestModelLoader()
        # Engine takes 0.25 seconds, but timeout is set to 0.06 seconds
        delayed_engine = PipelineTestInferenceEngine(delay_seconds=0.25)

        model = ml.load(
            model_source="local://models/slow_model",
            timeout_seconds=0.06,
            model_loader=loader,
            inference_engine=delayed_engine,
        )

        ctx = ExecutionContext(correlation_id="timeout-corr-101")
        result = model.process("Input that triggers timeout", context=ctx)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ExecutionTimeoutError")
        self.assertIn("Execution exceeded configured timeout", result.error)
        self.assertEqual(result.context.correlation_id, "timeout-corr-101")
        self.assertEqual(result.state, ExecutionState.FAILED)

        # Allow delayed worker thread to exit before teardown
        time.sleep(0.22)
        model.stop()


class TestPipelineResourceTeardown(unittest.TestCase):
    """
    7. Tests resource cleanup after successful execution and after failure.
    """

    def test_model_unload_lifecycle(self):
        """Consumer calls model.unload() to release model weights while keeping manager active."""
        loader = PipelineTestModelLoader()
        model = ml.load(model_source="local://models/v1", model_loader=loader)

        self.assertEqual(loader.load_count, 1)
        self.assertEqual(loader.unload_count, 0)
        self.assertEqual(model.state, ExecutionState.READY)

        # Unload
        model.unload()

        self.assertEqual(loader.unload_count, 1)
        self.assertEqual(model.state, ExecutionState.UNINITIALIZED)
        self.assertFalse(model.is_ready)

        # Executing while UNINITIALIZED raises StateError
        with self.assertRaises(StateError):
            model.process("cannot execute when uninitialized")

        model.stop()

    def test_model_stop_full_teardown(self):
        """Consumer calls model.stop() to terminate runtime and worker executor."""
        loader = PipelineTestModelLoader()
        model = ml.load(model_source="local://models/v1", model_loader=loader)

        model.stop()

        self.assertEqual(model.state, ExecutionState.STOPPED)
        self.assertFalse(model.is_ready)
        self.assertEqual(loader.unload_count, 1)

        # Repeated stop is idempotent
        model.stop()
        self.assertEqual(model.state, ExecutionState.STOPPED)

    def test_auto_cleanup_triggered_on_execution_failure(self):
        """When auto_cleanup is True, resource manager cleanup is triggered on stage failure."""
        loader = PipelineTestModelLoader()
        failing_engine = PipelineTestInferenceEngine(should_fail=True)
        resource_mgr = PipelineTestResourceManager()

        model = ml.load(
            model_source="local://models/v1",
            auto_cleanup=True,
            model_loader=loader,
            inference_engine=failing_engine,
            resource_manager=resource_mgr,
        )

        result = model.process("payload that causes failure")
        self.assertFalse(result.success)
        self.assertGreaterEqual(resource_mgr.cleanup_calls, 1)

        model.stop()


if __name__ == "__main__":
    unittest.main()
