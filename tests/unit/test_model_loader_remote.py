"""
Unit tests for MinIO / S3 remote model artifact retrieval in ModelLoader.

Verifies:
- URI scheme validation (s3://, minio://, seaweedfs://)
- Missing bucket or key validation
- Handling of non-existent remote artifacts
- Network and retrieval failure error propagation
- Multi-file directory prefix downloading (e.g., Hugging Face model folders)
- Archive unpacking (.tar.gz, .zip) and nested directory resolution
- Staging directory isolation and custom download_dir support
- Complete cleanup of temporary directories on load failure and on unload()
"""

import os
import sys
import json
import tarfile
import zipfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import boto3
    import botocore
except ImportError:
    mock_boto3 = MagicMock()
    mock_botocore = MagicMock()
    sys.modules["boto3"] = mock_boto3
    sys.modules["boto3.s3"] = mock_boto3.s3
    sys.modules["boto3.s3.transfer"] = mock_boto3.s3.transfer
    sys.modules["botocore"] = mock_botocore
    sys.modules["botocore.config"] = mock_botocore.config
    sys.modules["botocore.exceptions"] = mock_botocore.exceptions

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig, StorageConfig
from ml_framework.models.model_loader import ModelLoader
from ml_framework.errors import StorageError, ModelLoadError, ModelFormatError


class TestModelLoaderRemote(unittest.TestCase):
    """Unit test suite for ModelLoader remote artifact downloading and staging."""

    def setUp(self):
        self.storage_cfg = StorageConfig(
            endpoint_url="http://mock-minio:9000",
            access_key="mock-access",
            secret_key="mock-secret",
            secure=False,
            verify_ssl=False,
        )
        self.loader = ModelLoader(storage_config=self.storage_cfg)

    def tearDown(self):
        self.loader.close()

    def test_invalid_storage_scheme_rejected(self):
        """Test that non-storage schemes like http:// or gcs:// are rejected for remote downloading."""
        with self.assertRaises(StorageError) as cm:
            self.loader._download_remote_artifact("gcs://my-bucket/model")
        self.assertIn("Unsupported remote storage scheme", str(cm.exception))

    def test_missing_bucket_rejected(self):
        """Test that URIs missing bucket names are rejected."""
        with self.assertRaises(StorageError) as cm:
            self.loader._download_remote_artifact("s3:///only-key")
        self.assertIn("Missing bucket name", str(cm.exception))

    def test_missing_key_rejected(self):
        """Test that URIs missing object keys are rejected."""
        with self.assertRaises(StorageError) as cm:
            self.loader._download_remote_artifact("minio://bucket-only/")
        self.assertIn("Missing object key or prefix", str(cm.exception))

    def test_boto3_import_error(self):
        """Test that missing boto3 raises actionable StorageError."""
        with patch.dict(sys.modules, {"boto3": None}):
            with self.assertRaises(StorageError) as cm:
                self.loader._download_remote_artifact("s3://my-bucket/my-model")
            self.assertIn("boto3 is required", str(cm.exception))

    def test_remote_artifact_not_found(self):
        """Test that non-existent artifact in storage raises StorageError."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        # Return empty list of contents for both directory and exact prefix searches
        mock_paginator.paginate.return_value = [{"Contents": []}]
        mock_s3.get_paginator.return_value = mock_paginator

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(StorageError) as cm:
                self.loader._download_remote_artifact("s3://my-bucket/nonexistent_model")
            self.assertIn("not found in storage", str(cm.exception))

        # Ensure no temporary staging directories were leaked
        self.assertEqual(len(self.loader._temp_dirs), 0)

    def test_remote_retrieval_failure_and_cleanup(self):
        """Test that network/download failures clean up staging directory and raise StorageError."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "models/bert/config.json"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator
        mock_s3.download_file.side_effect = RuntimeError("Connection reset by peer")

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(StorageError) as cm:
                self.loader._download_remote_artifact("s3://my-bucket/models/bert")
            self.assertIn("Connection reset by peer", str(cm.exception))

        # Temporary staging directory must have been cleaned up
        self.assertEqual(len(self.loader._temp_dirs), 0)

    def test_download_directory_prefix_model_success(self):
        """Test downloading a complete multi-file model artifact directory from MinIO/S3."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()

        remote_keys = [
            {"Key": "models/my_hf_model/config.json"},
            {"Key": "models/my_hf_model/model.safetensors"},
            {"Key": "models/my_hf_model/tokenizer.json"},
            {"Key": "models/my_hf_model/vocab.txt"},
        ]
        mock_paginator.paginate.return_value = [{"Contents": remote_keys}]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download(Bucket, Key, Filename):
            os.makedirs(os.path.dirname(Filename), exist_ok=True)
            with open(Filename, "w") as f:
                if Key.endswith("config.json"):
                    f.write('{"model_type": "distilbert", "architectures": ["DistilBertForTokenClassification"]}')
                else:
                    f.write(f"content for {Key}")

        mock_s3.download_file.side_effect = mock_download

        with patch("boto3.client", return_value=mock_s3):
            staging_dir = self.loader._download_remote_artifact("s3://ml-models/models/my_hf_model")

            self.assertTrue(os.path.isdir(staging_dir))
            self.assertTrue(os.path.isfile(os.path.join(staging_dir, "config.json")))
            self.assertTrue(os.path.isfile(os.path.join(staging_dir, "model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(staging_dir, "tokenizer.json")))
            self.assertTrue(os.path.isfile(os.path.join(staging_dir, "vocab.txt")))

            # Verify format detector can read the staged model
            fw, arch, mtype = self.loader.format_detector.detect(staging_dir)
            self.assertEqual(fw, "transformers")
            self.assertEqual(arch, "DistilBertForTokenClassification")
            self.assertEqual(mtype, "distilbert")

    def test_download_archive_artifact_unpacking_and_cleanup(self):
        """Test downloading an archive artifact (.tar.gz) unpacks files and deletes the archive."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        # For an exact single file match, dir search is empty, exact search returns 1 object
        mock_paginator.paginate.side_effect = [
            [{"Contents": []}],  # dir_prefix
            [{"Contents": [{"Key": "models/spacy_model.tar.gz"}]}],  # exact key
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download_tar(Bucket, Key, Filename):
            # Create a real tar.gz file
            with tempfile.NamedTemporaryFile("wb", delete=False) as inner:
                inner.write(b"meta.json data")
                inner_path = inner.name
            try:
                with tarfile.open(Filename, "w:gz") as tar:
                    tar.add(inner_path, arcname="meta.json")
            finally:
                os.remove(inner_path)

        mock_s3.download_file.side_effect = mock_download_tar

        with patch("boto3.client", return_value=mock_s3):
            staging_dir = self.loader._download_remote_artifact("s3://models-bucket/models/spacy_model.tar.gz")

            self.assertTrue(os.path.isdir(staging_dir))
            self.assertTrue(os.path.isfile(os.path.join(staging_dir, "meta.json")))
            # The archive file itself must have been removed
            self.assertFalse(os.path.exists(os.path.join(staging_dir, "spacy_model.tar.gz")))

    def test_download_archive_nested_directory_resolution(self):
        """Test archive unpacking resolves to inner subdirectory if archive contains top-level folder."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.side_effect = [
            [{"Contents": []}],
            [{"Contents": [{"Key": "models/packaged_model.zip"}]}],
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download_zip(Bucket, Key, Filename):
            with zipfile.ZipFile(Filename, "w") as z:
                z.writestr("inner_folder/config.cfg", "[nlp]\nlang = 'en'")

        mock_s3.download_file.side_effect = mock_download_zip

        with patch("boto3.client", return_value=mock_s3):
            resolved_dir = self.loader._download_remote_artifact("minio://bucket/models/packaged_model.zip")

            self.assertTrue(os.path.isdir(resolved_dir))
            self.assertTrue(os.path.isfile(os.path.join(resolved_dir, "config.cfg")))
            self.assertTrue(resolved_dir.endswith("inner_folder"))

    def test_incomplete_model_cleanup_on_load_error(self):
        """Test that an incomplete remote model directory cleans up temporary staging on error."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "models/corrupt_model/config.json"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download(Bucket, Key, Filename):
            os.makedirs(os.path.dirname(Filename), exist_ok=True)
            with open(Filename, "w") as f:
                # Corrupted / invalid config.json that causes FormatDetector or json load to fail
                f.write('{ INVALID JSON CONTENT')

        mock_s3.download_file.side_effect = mock_download

        with patch("boto3.client", return_value=mock_s3):
            with self.assertRaises(ModelFormatError):
                self.loader.load("s3://bucket/models/corrupt_model")

        # Staging directory must have been completely cleaned up
        self.assertEqual(len(self.loader._temp_dirs), 0)

    def test_cleanup_on_unload(self):
        """Test that calling unload() cleans up all remote model staging directories."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"Contents": [{"Key": "models/valid_spacy/meta.json"}]}
        ]
        mock_s3.get_paginator.return_value = mock_paginator

        def mock_download(Bucket, Key, Filename):
            os.makedirs(os.path.dirname(Filename), exist_ok=True)
            with open(Filename, "w") as f:
                f.write('{"lang": "en", "name": "core_web_sm"}')

        mock_s3.download_file.side_effect = mock_download

        mock_adapter = MagicMock()
        mock_adapter.unload = MagicMock()
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_resolved.return_value = mock_adapter
        mock_loader_instance.load.return_value = mock_adapter
        mock_loader_cls = MagicMock(return_value=mock_loader_instance)

        with patch("boto3.client", return_value=mock_s3), \
             patch.object(self.loader.env_manager, "validate_dependencies"), \
             patch.object(self.loader, "_loader_registry", {"spacy": mock_loader_cls}):

            loaded_model = self.loader.load("s3://bucket/models/valid_spacy")
            self.assertEqual(len(self.loader._temp_dirs), 1)
            staged_path = self.loader._temp_dirs[0]
            self.assertTrue(os.path.exists(staged_path))

            # Unload model
            self.loader.unload(loaded_model)
            mock_adapter.unload.assert_called_once()

            # Verify staging directory removed from disk
            self.assertFalse(os.path.exists(staged_path))
            self.assertEqual(len(self.loader._temp_dirs), 0)

    def test_custom_download_dir_staging(self):
        """Test that configuring download_dir places staging directories in that custom path."""
        with tempfile.TemporaryDirectory() as custom_cache:
            custom_cfg = StorageConfig(
                endpoint_url="http://mock-minio:9000",
                access_key="mock-access",
                secret_key="mock-secret",
                download_dir=custom_cache,
            )
            custom_loader = ModelLoader(storage_config=custom_cfg)

            mock_s3 = MagicMock()
            mock_paginator = MagicMock()
            mock_paginator.paginate.return_value = [
                {"Contents": [{"Key": "models/cached_model/config.cfg"}]}
            ]
            mock_s3.get_paginator.return_value = mock_paginator

            def mock_download(Bucket, Key, Filename):
                os.makedirs(os.path.dirname(Filename), exist_ok=True)
                with open(Filename, "w") as f:
                    f.write("[nlp]\nlang = 'en'")

            mock_s3.download_file.side_effect = mock_download

            try:
                with patch("boto3.client", return_value=mock_s3):
                    staging_dir = custom_loader._download_remote_artifact("s3://bucket/models/cached_model")
                    self.assertTrue(staging_dir.startswith(custom_cache))
            finally:
                custom_loader.close()


if __name__ == "__main__":
    unittest.main()
