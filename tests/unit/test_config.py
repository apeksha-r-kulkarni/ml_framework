"""
Unit tests for configuration schemas and validation.
"""

import sys
import unittest
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config import FrameworkConfig, StorageConfig, InputConfig
from ml_framework.errors import ConfigurationError


class TestConfig(unittest.TestCase):
    """Test suite for FrameworkConfig and StorageConfig."""

    def test_default_framework_config(self):
        """Test default framework configuration values."""
        cfg = FrameworkConfig()
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.timeout_seconds, 30.0)
        self.assertEqual(cfg.device, "auto")
        self.assertEqual(cfg.framework, "auto")
        self.assertTrue(cfg.enable_telemetry)
        self.assertTrue(cfg.auto_cleanup)
        self.assertIsInstance(cfg.storage_config, StorageConfig)
        # Should validate without error
        cfg.validate()

    def test_from_dict_valid(self):
        """Test creating FrameworkConfig from dictionary."""
        data = {
            "model_name": "speech_lid",
            "model_version": "2.1.0",
            "model_source": "minio://models/lid.onnx",
            "device": "cpu",
            "batch_size": 4,
            "timeout_seconds": 15.5,
            "storage_config": {
                "endpoint_url": "http://minio:9000",
                "access_key": "minioadmin",
                "secret_key": "miniopassword",
                "secure": False,
            },
        }
        cfg = FrameworkConfig.from_dict(data)
        self.assertEqual(cfg.model_name, "speech_lid")
        self.assertEqual(cfg.model_version, "2.1.0")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.batch_size, 4)
        self.assertEqual(cfg.timeout_seconds, 15.5)
        self.assertEqual(cfg.storage_config.endpoint_url, "http://minio:9000")
        self.assertEqual(cfg.storage_config.access_key, "minioadmin")
        self.assertFalse(cfg.storage_config.secure)

    def test_invalid_batch_size(self):
        """Test that batch_size < 1 raises ConfigurationError."""
        cfg = FrameworkConfig(batch_size=0)
        with self.assertRaises(ConfigurationError) as cm:
            cfg.validate()
        self.assertIn("batch_size must be >= 1", str(cm.exception))

        cfg_neg = FrameworkConfig(batch_size=-5)
        with self.assertRaises(ConfigurationError):
            cfg_neg.validate()

    def test_invalid_timeout(self):
        """Test that timeout_seconds <= 0 raises ConfigurationError."""
        cfg = FrameworkConfig(timeout_seconds=0.0)
        with self.assertRaises(ConfigurationError) as cm:
            cfg.validate()
        self.assertIn("timeout_seconds must be > 0", str(cm.exception))

    def test_invalid_device(self):
        """Test that an unsupported device raises ConfigurationError."""
        cfg = FrameworkConfig(device="quantum_core")
        with self.assertRaises(ConfigurationError) as cm:
            cfg.validate()
        self.assertIn("Invalid device 'quantum_core'", str(cm.exception))

    def test_valid_cuda_index_device(self):
        """Test that cuda:<index> format is valid in FrameworkConfig."""
        cfg = FrameworkConfig(device="cuda:0")
        cfg.validate()
        self.assertEqual(cfg.device, "cuda:0")

    def test_from_dict_non_dict_input(self):
        """Test that non-dict input to from_dict raises ConfigurationError."""
        with self.assertRaises(ConfigurationError):
            FrameworkConfig.from_dict(["not", "a", "dict"])

    def test_storage_config_from_env_style_dict(self):
        """Test that StorageConfig parses environment variable style keys."""
        env_dict = {
            "MINIO_ENDPOINT": "http://192.168.1.100:9000",
            "MINIO_ACCESS_KEY": "testkey",
            "MINIO_SECRET_KEY": "testsecret",
            "MINIO_SECURE": "true",
            "MINIO_DOWNLOAD_DIR": "/tmp/artifacts",
        }
        storage = StorageConfig.from_dict(env_dict)
        self.assertEqual(storage.endpoint_url, "http://192.168.1.100:9000")
        self.assertEqual(storage.access_key, "testkey")
        self.assertEqual(storage.secret_key, "testsecret")
        self.assertTrue(storage.secure)
        self.assertEqual(storage.download_dir, "/tmp/artifacts")
        self.assertTrue(storage.verify_ssl)

    def test_storage_config_tls_verification_defaults_and_options(self):
        """Test default and explicit certificate verification settings for StorageConfig."""
        # 1. Default constructor preserves verify_ssl=True (secure default)
        default_cfg = StorageConfig()
        self.assertTrue(default_cfg.verify_ssl)

        # 2. from_dict without TLS ignore flag defaults to verify_ssl=True
        cfg_default = StorageConfig.from_dict({
            "MINIO_ENDPOINT": "https://minio.internal:9000",
            "MINIO_SECURE": "true",
        })
        self.assertTrue(cfg_default.secure)
        self.assertTrue(cfg_default.verify_ssl)

        # 3. Support MLFLOW_S3_IGNORE_TLS="true" or "1" (sets verify_ssl=False while keeping HTTPS)
        cfg_ignore_true = StorageConfig.from_dict({
            "MINIO_ENDPOINT": "https://minio.internal:9000",
            "MINIO_SECURE": "true",
            "MLFLOW_S3_IGNORE_TLS": "true",
        })
        self.assertTrue(cfg_ignore_true.secure)
        self.assertFalse(cfg_ignore_true.verify_ssl)

        cfg_ignore_one = StorageConfig.from_dict({
            "MLFLOW_S3_IGNORE_TLS": "1",
        })
        self.assertFalse(cfg_ignore_one.verify_ssl)

        # 4. Support MLFLOW_S3_IGNORE_TLS="false" preserves verify_ssl=True
        cfg_ignore_false = StorageConfig.from_dict({
            "MLFLOW_S3_IGNORE_TLS": "false",
        })
        self.assertTrue(cfg_ignore_false.verify_ssl)

        # 5. Support explicit verify_ssl boolean and custom CA bundle path
        cfg_verify_false = StorageConfig.from_dict({
            "verify_ssl": False,
        })
        self.assertFalse(cfg_verify_false.verify_ssl)

        cfg_ca_bundle = StorageConfig.from_dict({
            "verify_ssl": "/etc/ssl/certs/custom-ca.pem",
        })
        self.assertEqual(cfg_ca_bundle.verify_ssl, "/etc/ssl/certs/custom-ca.pem")

        # 6. Support framework-specific MINIO_VERIFY_SSL="false"
        cfg_minio_verify_false = StorageConfig.from_dict({
            "MINIO_VERIFY_SSL": "false",
        })
        self.assertFalse(cfg_minio_verify_false.verify_ssl)

        # 7. Support MINIO_CA_BUNDLE and AWS_CA_BUNDLE environment variables
        cfg_minio_ca = StorageConfig.from_dict({
            "MINIO_CA_BUNDLE": "/etc/ssl/certs/minio-ca.pem",
        })
        self.assertEqual(cfg_minio_ca.verify_ssl, "/etc/ssl/certs/minio-ca.pem")

        cfg_aws_ca = StorageConfig.from_dict({
            "AWS_CA_BUNDLE": "/etc/ssl/certs/aws-ca.pem",
        })
        self.assertEqual(cfg_aws_ca.verify_ssl, "/etc/ssl/certs/aws-ca.pem")

    def test_storage_validate_live_config(self):
        """Test validation of live StorageConfig with actionable errors."""
        # 1. Valid configuration
        storage = StorageConfig(
            endpoint_url="http://minio.team:9000",
            access_key="valid-access-key",
            secret_key="valid-secret-key"
        )
        storage.validate_live_config(bucket="test-bucket")

        # 2. Missing endpoint
        empty_storage = StorageConfig()
        with self.assertRaises(ConfigurationError) as cm:
            empty_storage.validate_live_config(bucket="test-bucket")
        self.assertIn("MINIO_ENDPOINT", str(cm.exception))

        # 3. Placeholder rejection
        placeholder_storage = StorageConfig(
            endpoint_url="http://minio.team:9000",
            access_key="your_minio_access_key_here",
            secret_key="your_minio_secret_key_here"
        )
        with self.assertRaises(ConfigurationError) as cm:
            placeholder_storage.validate_live_config(bucket="test-bucket")
        self.assertIn("Placeholder credentials detected", str(cm.exception))

    def test_input_validate_live_kafka_config(self):
        """Test validation of live InputConfig for Kafka."""
        from ml_framework.config import InputConfig

        # 1. Valid config
        cfg = InputConfig(
            kafka_bootstrap_servers="kafka.team:9092",
            kafka_topic="ml-framework-test-topic"
        )
        cfg.validate_live_kafka_config()

        # 2. Missing bootstrap server
        cfg_missing = InputConfig(kafka_topic="test-topic")
        with self.assertRaises(ConfigurationError) as cm:
            cfg_missing.validate_live_kafka_config()
        self.assertIn("KAFKA_BOOTSTRAP_SERVER", str(cm.exception))

        # 3. Invalid topic name with spaces
        cfg_invalid_topic = InputConfig(
            kafka_bootstrap_servers="localhost:9092",
            kafka_topic="bad topic with spaces!"
        )
        with self.assertRaises(ConfigurationError) as cm:
            cfg_invalid_topic.validate_live_kafka_config()
        self.assertIn("Invalid Kafka topic name", str(cm.exception))

    def test_kafka_ssl_cert_configuration(self):
        """Test that Kafka SSL certificate settings are loaded from dictionary/env."""
        from ml_framework.config import InputConfig
        data = {
            "KAFKA_BOOTSTRAP_SERVER": "kafka.secure:9093",
            "KAFKA_SOURCE_TOPIC": "secure-topic",
            "KAFKA_SECURITY_PROTOCOL": "SSL",
            "KAFKA_SSL_CA_LOCATION": "/certs/ca.pem",
            "KAFKA_SSL_CERTIFICATE_LOCATION": "/certs/cert.pem",
            "KAFKA_SSL_KEY_LOCATION": "/certs/key.pem",
        }
        cfg = InputConfig.from_dict(data)
        self.assertEqual(cfg.kafka_options.get("security.protocol"), "SSL")
        self.assertEqual(cfg.kafka_options.get("ssl.ca.location"), "/certs/ca.pem")
        self.assertEqual(cfg.kafka_options.get("ssl.certificate.location"), "/certs/cert.pem")
        self.assertEqual(cfg.kafka_options.get("ssl.key.location"), "/certs/key.pem")

    def test_storage_config_from_env_precedence(self):
        """Test configuration precedence: explicit env dict > os.environ > default values."""
        import os
        from unittest.mock import patch
        from ml_framework.config import parse_env_file

        mock_os_env = {
            "MINIO_ENDPOINT": "http://os-env-host:9000",
            "MINIO_ACCESS_KEY": "os-access",
            "MINIO_SECRET_KEY": "os-secret",
            "MINIO_SECURE": "false",
        }

        with patch.dict(os.environ, mock_os_env, clear=True):
            # 1. from_env resolves from os.environ
            cfg_os = StorageConfig.from_env(load_dotenv=False)
            self.assertEqual(cfg_os.endpoint_url, "http://os-env-host:9000")
            self.assertEqual(cfg_os.access_key, "os-access")
            self.assertEqual(cfg_os.secret_key, "os-secret")
            self.assertFalse(cfg_os.secure)

            # 2. Explicit env dict overrides os.environ
            custom_env = {
                "MINIO_ENDPOINT": "https://custom-host:9000",
                "MINIO_SECURE": "true",
            }
            cfg_override = StorageConfig.from_env(env=custom_env, load_dotenv=False)
            self.assertEqual(cfg_override.endpoint_url, "https://custom-host:9000")
            self.assertEqual(cfg_override.access_key, "os-access")
            self.assertTrue(cfg_override.secure)

    def test_parse_env_file_isolated(self):
        """Test that parse_env_file parses key-value pairs without mutating os.environ."""
        import os
        import tempfile
        from ml_framework.config import parse_env_file

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# Sample comment\n")
            f.write("TEST_CONFIG_KEY=custom_value\n")
            f.write("QUOTED_KEY=\"quoted_value\"\n")
            temp_env_path = f.name

        try:
            self.assertNotIn("TEST_CONFIG_KEY", os.environ)
            parsed = parse_env_file(temp_env_path)
            self.assertEqual(parsed.get("TEST_CONFIG_KEY"), "custom_value")
            self.assertEqual(parsed.get("QUOTED_KEY"), "quoted_value")
            # Global os.environ must NOT be modified
            self.assertNotIn("TEST_CONFIG_KEY", os.environ)
        finally:
            if os.path.exists(temp_env_path):
                os.remove(temp_env_path)

    def test_from_dict_isolation_from_os_environ(self):
        """Test that StorageConfig.from_dict ignores unrelated os.environ variables."""
        import os
        from unittest.mock import patch

        polluted_env = {
            "MINIO_ENDPOINT": "http://polluted-host:9000",
            "MINIO_SECURE": "true",
            "MLFLOW_S3_IGNORE_TLS": "true",
        }

        with patch.dict(os.environ, polluted_env):
            # from_dict with empty dict returns defaults without inheriting from polluted os.environ
            cfg = StorageConfig.from_dict({})
            self.assertIsNone(cfg.endpoint_url)
            self.assertFalse(cfg.secure)
            self.assertTrue(cfg.verify_ssl)

            # from_dict with explicit values preserves them
            cfg_explicit = StorageConfig.from_dict({"endpoint_url": "http://clean-host:9000", "secure": False})
            self.assertEqual(cfg_explicit.endpoint_url, "http://clean-host:9000")
            self.assertFalse(cfg_explicit.secure)
            self.assertTrue(cfg_explicit.verify_ssl)

    def test_framework_config_flat_dict_ingestion(self):
        """Test FrameworkConfig.from_dict automatically maps top-level storage and kafka keys."""
        flat_dict = {
            "MINIO_ENDPOINT": "https://192.168.2.175:9000",
            "MINIO_ACCESS_KEY": "test_access",
            "MINIO_SECRET_KEY": "test_secret",
            "MINIO_CA_BUNDLE": "/etc/ssl/certs/minio.crt",
            "KAFKA_BOOTSTRAP_SERVER": "192.168.2.175:9092",
            "KAFKA_SOURCE_TOPIC": "input-events",
            "model_name": "conformer_ctc",
            "task": "transcription",
        }
        cfg = FrameworkConfig.from_dict(flat_dict)
        self.assertEqual(cfg.storage_config.endpoint_url, "https://192.168.2.175:9000")
        self.assertEqual(cfg.storage_config.access_key, "test_access")
        self.assertEqual(cfg.storage_config.verify_ssl, "/etc/ssl/certs/minio.crt")
        self.assertEqual(cfg.input_config.kafka_bootstrap_servers, "192.168.2.175:9092")
        self.assertEqual(cfg.input_config.kafka_topic, "input-events")

    def test_from_env_methods(self):
        """Test from_env methods on InputConfig and FrameworkConfig."""
        mock_env = {
            "KAFKA_BOOTSTRAP_SERVER": "broker:9092",
            "KAFKA_TEST_TOPIC": "test-topic",
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESS_KEY": "key1",
            "MINIO_SECRET_KEY": "sec1",
        }
        in_cfg = InputConfig.from_env(env=mock_env, load_dotenv=False)
        self.assertEqual(in_cfg.kafka_bootstrap_servers, "broker:9092")
        self.assertEqual(in_cfg.kafka_topic, "test-topic")

        fw_cfg = FrameworkConfig.from_env(env=mock_env, load_dotenv=False)
        self.assertEqual(fw_cfg.input_config.kafka_bootstrap_servers, "broker:9092")
        self.assertEqual(fw_cfg.storage_config.endpoint_url, "http://minio:9000")


if __name__ == "__main__":
    unittest.main()

