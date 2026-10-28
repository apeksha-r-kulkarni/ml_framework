"""
Offline Unit Tests for MinIO-to-Triton Model Retrieval and Staging in ML Framework.

Verifies:
1. Configurable MinIO model selection (model_bucket and model_key resolution).
2. Remote artifact download from MinIO via mocked Boto3 client.
3. Native .nemo archive preservation (ensuring archive is not extracted or removed).
4. Staging of downloaded artifacts into Triton staging directories.
5. Format and backend compatibility validation (NeMo Python backend vs ONNX).
6. Clear error handling:
   - Authentication failure (403, InvalidAccessKeyId) -> StorageError
   - Missing object or bucket (404, NoSuchKey) -> StorageError
   - Endpoint connectivity failure (EndpointConnectionError) -> StorageError
7. Exclusive production inference execution via Triton without local adapter fallback.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import boto3
    import botocore
    from botocore.exceptions import ClientError, EndpointConnectionError
except ImportError:
    mock_boto3 = MagicMock()
    mock_botocore = MagicMock()
    sys.modules["boto3"] = mock_boto3
    sys.modules["boto3.s3"] = mock_boto3.s3
    sys.modules["boto3.s3.transfer"] = mock_boto3.s3.transfer
    sys.modules["botocore"] = mock_botocore
    sys.modules["botocore.config"] = mock_botocore.config
    sys.modules["botocore.exceptions"] = mock_botocore.exceptions
    ClientError = Exception
    EndpointConnectionError = Exception

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework import ml, FrameworkConfig, StorageConfig, TritonConfig, ExecutionState
from ml_framework.models.model_loader import ModelLoader
from ml_framework.inference.triton_engine import TritonInferenceEngine
from ml_framework.inference.triton_client import TritonClient
from ml_framework.errors import StorageError, ModelLoadError, ModelFormatError, InferenceError


class TestMinIOToTritonRetrieval(unittest.TestCase):
    """Test suite for MinIO-to-Triton model retrieval, validation, and staging."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.staging_dir = os.path.join(self.temp_dir.name, "triton_staging")
        os.makedirs(self.staging_dir, exist_ok=True)

        self.storage_cfg = StorageConfig(
            endpoint_url="http://mock-minio:9000",
            access_key="mock-access-key",
            secret_key="mock-secret-key",
            model_bucket="speech-models",
            model_key="conformer/stt_en_conformer_ctc_large.nemo",
        )
        self.triton_cfg = TritonConfig(
            endpoint_url="http://localhost:8010",
            model_name="nemo_conformer_python",
            backend="python",
            staging_dir=self.staging_dir,
            check_model_ready=False,
        )
        self.loader = ModelLoader(storage_config=self.storage_cfg)

    def tearDown(self):
        self.loader.close()
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Configurable MinIO Bucket and Object Key Resolution
    # -------------------------------------------------------------------------

    def test_configurable_model_bucket_and_key_in_framework_config(self):
        """Test configuring model via model_bucket and model_key resolves to s3 URI."""
        cfg = FrameworkConfig(
            model_bucket="my-bucket",
            model_key="path/to/model.nemo",
        )
        self.assertEqual(cfg.model_source, "s3://my-bucket/path/to/model.nemo")
        self.assertEqual(cfg.model_bucket, "my-bucket")
        self.assertEqual(cfg.model_key, "path/to/model.nemo")

    def test_configurable_model_bucket_and_key_from_dict(self):
        """Test configuring model via dictionary with MINIO_MODEL_BUCKET and MINIO_MODEL_KEY."""
        cfg = FrameworkConfig.from_dict({
            "MINIO_MODEL_BUCKET": "team-speech",
            "MINIO_MODEL_KEY": "asr/conformer.nemo",
        })
        self.assertEqual(cfg.model_source, "s3://team-speech/asr/conformer.nemo")
        self.assertEqual(cfg.model_bucket, "team-speech")
        self.assertEqual(cfg.model_key, "asr/conformer.nemo")

    def test_storage_config_does_not_reveal_secrets_in_repr(self):
        """Test StorageConfig never prints secret_key in repr or str."""
        cfg = StorageConfig(
            endpoint_url="http://minio:9000",
            access_key="admin",
            secret_key="SUPER_SECRET_VALUE_12345",
        )
        cfg_repr = repr(cfg)
        self.assertNotIn("SUPER_SECRET_VALUE_12345", cfg_repr)

    # -------------------------------------------------------------------------
    # 2. MinIO Retrieval and Intact .nemo Preservation
    # -------------------------------------------------------------------------

    def test_nemo_archive_preserved_intact_and_staged_for_triton(self):
        """
        Verify that downloading a .nemo archive from MinIO keeps the file intact
        (does NOT unpack into a directory, does NOT delete .nemo), and stages it to Triton.
        """
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = [
            [{"Contents": []}],  # prefix search
            [{"Contents": [{"Key": "conformer/stt_en_conformer_ctc_large.nemo"}]}],  # exact key
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        # Simulate downloading a valid .nemo archive file
        def mock_download(Bucket, Key, Filename):
            os.makedirs(os.path.dirname(Filename), exist_ok=True)
            with open(Filename, "wb") as f:
                f.write(b"NEMO_ARCHIVE_BINARY_DATA_DUMMY")

        mock_s3.download_file.side_effect = mock_download

        with patch("boto3.client", return_value=mock_s3):
            # Call load_for_triton with MinIO S3 source
            metadata = self.loader.load_for_triton(
                model_source="s3://speech-models/conformer/stt_en_conformer_ctc_large.nemo",
                config=FrameworkConfig(
                    model_bucket="speech-models",
                    model_key="conformer/stt_en_conformer_ctc_large.nemo",
                    triton_config=self.triton_cfg,
                ),
            )

            # 1. Metadata verification
            self.assertEqual(metadata["model_name"], "nemo_conformer_python")
            self.assertEqual(metadata["framework"], "nemo")
            self.assertEqual(metadata["triton_backend"], "python")
            self.assertTrue(metadata["is_triton_served"])

            # 2. Staging verification
            staged_path = metadata["staged_path"]
            self.assertTrue(os.path.isfile(staged_path))
            self.assertTrue(staged_path.endswith("stt_en_conformer_ctc_large.nemo"))
            self.assertTrue(staged_path.startswith(self.staging_dir))

            # 3. Content verification (intact file, not unpacked)
            with open(staged_path, "rb") as f:
                content = f.read()
            self.assertEqual(content, b"NEMO_ARCHIVE_BINARY_DATA_DUMMY")

    # -------------------------------------------------------------------------
    # 3. Format and Triton Backend Compatibility Validation
    # -------------------------------------------------------------------------

    def test_nemo_python_backend_rejects_onnx_model(self):
        """Test that passing an ONNX model to Triton NeMo Python backend raises ModelFormatError."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_onnx:
            tmp_onnx.write(b"ONNX_MODEL_BINARY")
            onnx_path = tmp_onnx.name

        try:
            with self.assertRaises(ModelFormatError) as cm:
                self.loader.validate_backend_compatibility(
                    local_path=onnx_path,
                    backend="python",
                    model_name="nemo_conformer_python",
                )
            self.assertIn("incompatible with Triton backend 'python'", str(cm.exception))
            self.assertIn("requires a native NeMo model checkpoint (.nemo)", str(cm.exception))
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def test_nemo_python_backend_rejects_transformers_text_model(self):
        """Test that passing a HuggingFace Bert model to Triton NeMo Python backend raises ModelFormatError."""
        model_dir = os.path.join(self.temp_dir.name, "bert_model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            f.write('{"model_type": "bert", "architectures": ["BertForSequenceClassification"]}')

        with self.assertRaises(ModelFormatError) as cm:
            self.loader.validate_backend_compatibility(
                local_path=model_dir,
                backend="python",
                model_name="nemo_conformer_python",
            )
        self.assertIn("incompatible with Triton backend 'python'", str(cm.exception))

    def test_onnx_backend_accepts_onnx_and_rejects_nemo(self):
        """Test that ONNX backend accepts .onnx and rejects .nemo."""
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp_onnx:
            tmp_onnx.write(b"ONNX_DATA")
            onnx_path = tmp_onnx.name

        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as tmp_nemo:
            tmp_nemo.write(b"NEMO_DATA")
            nemo_path = tmp_nemo.name

        try:
            # ONNX model with ONNX backend succeeds
            fw, arch, mtype = self.loader.validate_backend_compatibility(
                local_path=onnx_path,
                backend="onnxruntime",
            )
            self.assertEqual(fw, "onnx")

            # NeMo model with ONNX backend fails
            with self.assertRaises(ModelFormatError) as cm:
                self.loader.validate_backend_compatibility(
                    local_path=nemo_path,
                    backend="onnxruntime",
                )
            self.assertIn("incompatible with Triton backend 'onnxruntime'", str(cm.exception))
        finally:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            if os.path.exists(nemo_path):
                os.remove(nemo_path)

    # -------------------------------------------------------------------------
    # 4. Error Handling: Authentication, Missing Object, Connectivity
    # -------------------------------------------------------------------------

    def test_minio_authentication_failure_raises_clear_storage_error(self):
        """Test that 403 / AccessDenied from MinIO raises clear StorageError without revealing secrets."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()

        # Simulate Boto3 ClientError 403 / AccessDenied
        err_response = {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}
        if ClientError is not Exception:
            client_error = ClientError(err_response, "ListObjectsV2")
        else:
            client_error = RuntimeError("AccessDenied: Access Denied")
            client_error.response = err_response

        mock_paginator.paginate.side_effect = client_error
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(StorageError) as cm:
                self.loader.load_for_triton(
                    model_source="s3://speech-models/conformer/stt_en_conformer_ctc_large.nemo",
                    config=FrameworkConfig(
                        model_bucket="speech-models",
                        model_key="conformer/stt_en_conformer_ctc_large.nemo",
                        storage_config=self.storage_cfg,
                        triton_config=self.triton_cfg,
                    ),
                )

            err_str = str(cm.exception)
            self.assertIn("Authentication failed for MinIO storage", err_str)
            self.assertIn("verify MINIO_ACCESS_KEY and MINIO_SECRET_KEY", err_str)
            self.assertNotIn("mock-secret-key", err_str)

    def test_minio_missing_object_raises_clear_storage_error(self):
        """Test that non-existent model artifact in MinIO bucket raises actionable StorageError."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        # Both directory search and exact search return empty
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(StorageError) as cm:
                self.loader.load_for_triton(
                    model_source="s3://speech-models/missing/nonexistent_model.nemo",
                    config=FrameworkConfig(
                        model_bucket="speech-models",
                        model_key="missing/nonexistent_model.nemo",
                        storage_config=self.storage_cfg,
                        triton_config=self.triton_cfg,
                    ),
                )

            err_str = str(cm.exception)
            self.assertIn("Model artifact not found in storage", err_str)
            self.assertIn("speech-models", err_str)

    def test_minio_endpoint_connection_failure_raises_clear_storage_error(self):
        """Test that MinIO endpoint connectivity failure raises actionable StorageError."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()

        if EndpointConnectionError is not Exception:
            conn_err = EndpointConnectionError(endpoint_url="http://mock-minio:9000")
        else:
            conn_err = ConnectionError("Could not connect to the endpoint URL: http://mock-minio:9000")

        mock_paginator.paginate.side_effect = conn_err
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(StorageError) as cm:
                self.loader.load_for_triton(
                    model_source="s3://speech-models/conformer/stt_en_conformer_ctc_large.nemo",
                    config=FrameworkConfig(
                        model_bucket="speech-models",
                        model_key="conformer/stt_en_conformer_ctc_large.nemo",
                        storage_config=self.storage_cfg,
                        triton_config=self.triton_cfg,
                    ),
                )

            err_str = str(cm.exception)
            self.assertIn("Failed to connect to MinIO storage endpoint", err_str)

    # -------------------------------------------------------------------------
    # 5. End-to-End Orchestration via ml.load with MinIO Source
    # -------------------------------------------------------------------------

    def test_ml_load_with_minio_bucket_and_key_stages_for_triton(self):
        """
        Verify that ml.load(model_bucket=..., model_key=...) downloads from MinIO,
        stages to Triton staging directory, verifies Triton readiness, and reaches READY.
        """
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = [
            [{"Contents": []}],
            [{"Contents": [{"Key": "conformer/stt_en_conformer_ctc_large.nemo"}]}],
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download(Bucket, Key, Filename):
            os.makedirs(os.path.dirname(Filename), exist_ok=True)
            with open(Filename, "wb") as f:
                f.write(b"NEMO_CHECKPOINT_DATA")

        mock_s3.download_file.side_effect = mock_download

        mock_triton_client = MagicMock(spec=TritonClient)
        mock_triton_client.is_server_ready.return_value = True
        mock_triton_client.is_model_ready.return_value = True
        mock_triton_engine = TritonInferenceEngine(
            config=self.triton_cfg,
            client=mock_triton_client,
        )

        cfg_triton = TritonConfig(
            endpoint_url="http://localhost:8010",
            model_name="nemo_conformer_python",
            backend="python",
            staging_dir=self.staging_dir,
            check_model_ready=True,
        )
        cfg = FrameworkConfig(
            model_bucket="speech-models",
            model_key="conformer/stt_en_conformer_ctc_large.nemo",
            storage_config=self.storage_cfg,
            triton_config=cfg_triton,
        )

        from ml_framework.environment.environment_manager import EnvironmentManager
        mock_env_mgr = EnvironmentManager(override_packages={"boto3": True, "botocore": True})

        with patch("boto3.client", return_value=mock_s3):
            manager = ml.load(
                config=cfg,
                env_manager=mock_env_mgr,
                inference_engine=mock_triton_engine,
            )

            self.assertTrue(manager.is_ready)
            self.assertEqual(manager.state, ExecutionState.READY)
            self.assertEqual(manager.model_id, "nemo_conformer_python")

            # Check that the artifact was staged in self.staging_dir
            staged_file = os.path.join(self.staging_dir, "stt_en_conformer_ctc_large.nemo")
            self.assertTrue(os.path.isfile(staged_file))

            # Verify Triton readiness check was called
            mock_triton_client.is_server_ready.assert_called()
            mock_triton_client.is_model_ready.assert_called_with("nemo_conformer_python", "")

            manager.unload()
            manager.stop()

    def test_exclusive_triton_inference_execution_no_fallback(self):
        """
        Verify that inference forward pass is dispatched exclusively to TritonInferenceEngine,
        and never attempts direct local execution.
        """
        mock_triton_client = MagicMock(spec=TritonClient)
        mock_triton_client.is_server_ready.return_value = True
        mock_triton_client.is_model_ready.return_value = True
        mock_triton_client.infer.return_value = {
            "model_name": "nemo_conformer_python",
            "outputs": [
                {
                    "name": "TRANSCRIPT",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello triton world"],
                }
            ],
        }

        mock_triton_engine = TritonInferenceEngine(
            config=self.triton_cfg,
            client=mock_triton_client,
        )

        with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as tmp_model:
            tmp_model.write(b"NEMO_BINARY")
            model_path = tmp_model.name

        try:
            cfg = FrameworkConfig(
                model_source=model_path,
                framework="nemo",
                triton_config=self.triton_cfg,
            )

            manager = ml.load(
                config=cfg,
                inference_engine=mock_triton_engine,
            )

            self.assertTrue(manager.is_ready)
            result = manager.execute([0.0, 0.1, -0.1])
            self.assertTrue(result.success)
            self.assertEqual(result.output.get("transcription"), "hello triton world")

            # Verify TritonClient.infer was called
            mock_triton_client.infer.assert_called_once()

            manager.unload()
            manager.stop()
        finally:
            if os.path.exists(model_path):
                os.remove(model_path)

    def test_empty_model_bucket_and_key_raises_load_error(self):
        """Test that load_for_triton raises ModelLoadError if no source, bucket, or key is provided."""
        with self.assertRaises(ModelLoadError) as cm:
            self.loader.load_for_triton(
                model_source=None,
                config=FrameworkConfig(model_source=None),
            )
        self.assertIn("Model source cannot be empty", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
