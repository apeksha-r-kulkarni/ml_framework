"""
Unit tests for the NVIDIA Triton Inference Subsystem in ML Framework.

Verifies:
1. TritonConfig defaults, validation, and from_dict / from_env parsing.
2. TensorMapper audio decoding, tensor formatting, non-finite checks, and response parsing.
3. TritonClient health probes, readiness checks, inference requests, and retry/timeout handling.
4. TritonInferenceEngine readiness verification, model resolution, and forward pass execution.
5. ExecutionManager delegation exclusively to Triton (blocking direct model adapter predictions).
6. ml.load facade integration with TritonInferenceEngine.
"""

import io
import sys
import json
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig, TritonConfig
from ml_framework.errors import ConfigurationError, InferenceError
from ml_framework.results.execution_result import ExecutionContext, ExecutionState
from ml_framework.inference.triton_client import TritonClient
from ml_framework.inference.tensor_mapper import TensorMapper
from ml_framework.inference.triton_engine import TritonInferenceEngine
from ml_framework.execution_manager import ExecutionManager
from ml_framework import ml


class TestTritonInferenceEngine(unittest.TestCase):
    """Test suite for Triton configuration, client, mapper, engine, and framework delegation."""

    def test_triton_config_defaults_and_validation(self):
        """Test TritonConfig default values and boundary validation."""
        cfg = TritonConfig()
        self.assertEqual(cfg.endpoint_url, "http://localhost:8010")
        self.assertEqual(cfg.protocol, "http")
        self.assertEqual(cfg.model_version, "")
        self.assertEqual(cfg.timeout_seconds, 30.0)
        self.assertTrue(cfg.check_model_ready)
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.retry_interval_seconds, 0.5)
        cfg.validate()

        # Prepending http:// when missing
        cfg_no_scheme = TritonConfig(endpoint_url="triton-server:8000")
        self.assertEqual(cfg_no_scheme.endpoint_url, "http://triton-server:8000")

        # Empty endpoint_url
        with self.assertRaises(ConfigurationError):
            TritonConfig(endpoint_url="").validate()

        # Invalid protocol
        with self.assertRaises(ConfigurationError):
            TritonConfig(protocol="websocket").validate()

        # Non-positive timeout
        with self.assertRaises(ConfigurationError):
            TritonConfig(timeout_seconds=0.0).validate()

        # Negative retries
        with self.assertRaises(ConfigurationError):
            TritonConfig(max_retries=-1).validate()

    def test_triton_config_from_dict_and_env(self):
        """Test constructing TritonConfig from dict and environment variables."""
        data = {
            "TRITON_SERVER_URL": "http://remote-triton:8000",
            "TRITON_MODEL_NAME": "nemo_conformer_python",
            "TRITON_MODEL_VERSION": "1",
            "TRITON_TIMEOUT": "45.0",
            "TRITON_CHECK_READY": "false",
            "TRITON_MAX_RETRIES": "5",
            "input_tensor_map": {"audio": "AUDIO_SIGNAL"},
        }
        cfg = TritonConfig.from_dict(data)
        self.assertEqual(cfg.endpoint_url, "http://remote-triton:8000")
        self.assertEqual(cfg.model_name, "nemo_conformer_python")
        self.assertEqual(cfg.model_version, "1")
        self.assertEqual(cfg.timeout_seconds, 45.0)
        self.assertFalse(cfg.check_model_ready)
        self.assertEqual(cfg.max_retries, 5)
        self.assertEqual(cfg.input_tensor_map, {"audio": "AUDIO_SIGNAL"})

        # FrameworkConfig nested triton_config resolution
        fw_cfg = FrameworkConfig.from_dict({
            "model_name": "speech_model",
            "triton_config": {"endpoint_url": "http://localhost:8010"},
        })
        self.assertEqual(fw_cfg.triton_config.model_name, "speech_model")
        self.assertEqual(fw_cfg.triton_config.endpoint_url, "http://localhost:8010")

    def test_tensor_mapper_audio_ndarray_mapping(self):
        """Test TensorMapper correctly formats NumPy audio signals into Triton FP32 and INT32 tensors."""
        mapper = TensorMapper(default_sample_rate=16000)
        audio_signal = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

        inputs, outputs = mapper.build_request(audio_signal, model_name="nemo_conformer_python")
        self.assertEqual(len(inputs), 2)

        sig_tensor = next(t for t in inputs if t["name"] == "AUDIO_SIGNAL")
        self.assertEqual(sig_tensor["datatype"], "FP32")
        self.assertEqual(sig_tensor["shape"], [4])
        self.assertAlmostEqual(sig_tensor["data"][0], 0.1, places=5)

        sr_tensor = next(t for t in inputs if t["name"] == "SAMPLE_RATE")
        self.assertEqual(sr_tensor["datatype"], "INT32")
        self.assertEqual(sr_tensor["shape"], [1])
        self.assertEqual(sr_tensor["data"], [16000])

        self.assertEqual(outputs, [{"name": "TRANSCRIPT"}])

    def test_tensor_mapper_audio_non_finite_and_empty_checks(self):
        """Test TensorMapper rejects NaN/Inf audio samples and empty arrays."""
        mapper = TensorMapper()

        # NaN / Inf audio
        nan_audio = np.array([0.1, np.nan, 0.3], dtype=np.float32)
        with self.assertRaises(InferenceError) as cm:
            mapper.build_request(nan_audio, model_name="nemo_conformer_python")
        self.assertIn("non-finite", str(cm.exception).lower())

        inf_audio = np.array([0.1, np.inf, 0.3], dtype=np.float32)
        with self.assertRaises(InferenceError) as cm:
            mapper.build_request(inf_audio, model_name="nemo_conformer_python")
        self.assertIn("non-finite", str(cm.exception).lower())

        # Empty audio
        with self.assertRaises(InferenceError) as cm:
            mapper.build_request(np.array([], dtype=np.float32), model_name="nemo_conformer_python")
        self.assertIn("empty", str(cm.exception).lower())

    def test_tensor_mapper_speech_response_parsing(self):
        """Test TensorMapper decodes Triton TRANSCRIPT outputs into standard speech domain structure."""
        mapper = TensorMapper()
        triton_resp = {
            "model_name": "nemo_conformer_python",
            "model_version": "1",
            "outputs": [
                {
                    "name": "TRANSCRIPT",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["testing one two three"],
                }
            ],
        }
        res = mapper.parse_response(triton_resp)
        self.assertIsInstance(res, dict)
        self.assertEqual(res["task"], "transcription")
        self.assertEqual(res["text"], "testing one two three")
        self.assertEqual(res["transcription"], "testing one two three")

    def test_tensor_mapper_generic_dictionary_and_text(self):
        """Test TensorMapper handles general dict inputs and string text inputs."""
        mapper = TensorMapper(input_tensor_map={"raw_text": "INPUT_TEXT"})

        # String text input
        inputs, outputs = mapper.build_request("Hello world sentence")
        self.assertEqual(inputs[0]["datatype"], "BYTES")
        self.assertEqual(inputs[0]["data"], ["Hello world sentence"])

        # Dict input with mapping
        inputs, outputs = mapper.build_request({"raw_text": "mapped input"})
        self.assertEqual(inputs[0]["name"], "INPUT_TEXT")
        self.assertEqual(inputs[0]["data"], ["mapped input"])

    def test_triton_client_health_probes(self):
        """Test TritonClient liveness and readiness probe methods."""
        client = TritonClient(endpoint_url="http://mock-triton:8000")

        # Mock successful 200 responses
        mock_resp = MagicMock()
        mock_resp.getcode.return_value = 200
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            self.assertTrue(client.is_server_live())
            self.assertTrue(client.is_server_ready())
            self.assertTrue(client.is_model_ready("nemo_conformer_python", "1"))

        # Mock connection error
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            self.assertFalse(client.is_server_live())
            self.assertFalse(client.is_server_ready())
            self.assertFalse(client.is_model_ready("nemo_conformer_python"))

    def test_triton_client_error_and_retry_handling(self):
        """Test that client retries on transient errors and halts on 4xx client errors."""
        client = TritonClient(
            endpoint_url="http://mock-triton:8000",
            max_retries=2,
            retry_interval_seconds=0.01,
        )

        # 400 Bad Request should not retry
        http_400 = urllib.error.HTTPError(
            url="http://mock-triton:8000/v2/models/model/infer",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=io.BytesIO(b'{"error": "invalid tensor shape"}')
        )
        with patch("urllib.request.urlopen", side_effect=http_400) as mock_open:
            with self.assertRaises(InferenceError) as cm:
                client.infer("model", [{"name": "IN", "shape": [1], "datatype": "FP32", "data": [1.0]}])
            self.assertIn("HTTP 400", str(cm.exception))
            # Must fail immediately without retrying client errors
            self.assertEqual(mock_open.call_count, 1)

        # Connection error should retry max_retries + 1 times
        with patch("urllib.request.urlopen", side_effect=OSError("Connection refused")) as mock_open:
            with self.assertRaises(InferenceError) as cm:
                client.infer("model", [{"name": "IN", "shape": [1], "datatype": "FP32", "data": [1.0]}])
            self.assertIn("Failed to communicate with Triton", str(cm.exception))
            self.assertEqual(mock_open.call_count, 3)

    def test_triton_engine_verify_readiness(self):
        """Test TritonInferenceEngine readiness verification logic."""
        cfg = TritonConfig(endpoint_url="http://mock-triton:8000", model_name="nemo_conformer_python")
        mock_client = MagicMock(spec=TritonClient)

        # 1. Server not ready
        mock_client.is_server_ready.return_value = False
        engine = TritonInferenceEngine(config=cfg, client=mock_client)
        with self.assertRaises(InferenceError) as cm:
            engine.verify_readiness()
        self.assertIn("not ready or unreachable", str(cm.exception))

        # 2. Server ready but model not ready
        mock_client.is_server_ready.return_value = True
        mock_client.is_model_ready.return_value = False
        with self.assertRaises(InferenceError) as cm:
            engine.verify_readiness()
        self.assertIn("is not READY on Triton", str(cm.exception))

        # 3. Both server and model ready
        mock_client.is_model_ready.return_value = True
        self.assertTrue(engine.verify_readiness())

    def test_triton_engine_infer_success_and_metadata(self):
        """Test end-to-end TritonInferenceEngine.infer call with execution context metadata recording."""
        cfg = TritonConfig(
            endpoint_url="http://mock-triton:8000",
            model_name="nemo_conformer_python",
            model_version="1",
        )
        mock_client = MagicMock(spec=TritonClient)
        mock_client.infer.return_value = {
            "model_name": "nemo_conformer_python",
            "model_version": "1",
            "outputs": [
                {
                    "name": "TRANSCRIPT",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello from triton"],
                }
            ],
        }

        engine = TritonInferenceEngine(config=cfg, client=mock_client)
        ctx = ExecutionContext(correlation_id="corr-123", metadata={"model_name": "nemo_conformer_python"})

        result = engine.infer(
            model=None,
            preprocessed_input=np.array([0.05, -0.05], dtype=np.float32),
            context=ctx,
        )

        self.assertEqual(result["text"], "hello from triton")
        self.assertEqual(ctx.metadata["triton_endpoint"], "http://mock-triton:8000")
        self.assertEqual(ctx.metadata["triton_model_name"], "nemo_conformer_python")
        self.assertEqual(ctx.metadata["triton_model_version"], "1")
        self.assertIn("triton_inference_latency_ms", ctx.metadata)

    def test_execution_manager_blocks_direct_adapter_inference(self):
        """Test that ExecutionManager strictly blocks direct model adapter .predict() calls."""
        cfg = FrameworkConfig(model_source="local://dummy_model")

        # Mock adapter with predict method
        mock_adapter = MagicMock()
        mock_adapter.predict = MagicMock(return_value="direct_prediction_leak")

        mock_loader = MagicMock()
        mock_loader.load.return_value = mock_adapter

        # Manager without inference_engine
        manager = ExecutionManager(
            config=cfg,
            model_loader=mock_loader,
            inference_engine=None,
        )
        manager.initialize()

        # Executing inference must raise InferenceError and NEVER call mock_adapter.predict()
        res = manager.execute("test payload")
        self.assertFalse(res.success)
        self.assertEqual(res.error_type, "InferenceError")
        self.assertIn("Direct local execution via model adapter is disabled", res.error)
        mock_adapter.predict.assert_not_called()

    def test_ml_load_provisions_triton_engine_by_default(self):
        """Test that ml.load automatically provisions TritonInferenceEngine as the exclusive engine."""
        cfg = FrameworkConfig(
            model_name="nemo_conformer_python",
            triton_config=TritonConfig(
                endpoint_url="http://mock-triton:8000",
                check_model_ready=False,
            )
        )

        with patch("ml_framework.inference.triton_client.TritonClient.is_server_ready", return_value=True), \
             patch("ml_framework.inference.triton_client.TritonClient.is_model_ready", return_value=True):
            manager = ml.load(config=cfg)
            self.assertIsInstance(manager.inference_engine, TritonInferenceEngine)
            self.assertEqual(manager.inference_engine.config.model_name, "nemo_conformer_python")
            manager.stop()


if __name__ == "__main__":
    unittest.main()
