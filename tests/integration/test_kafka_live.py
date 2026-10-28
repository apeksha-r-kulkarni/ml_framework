"""
Opt-in Live Integration Tests for Kafka Message Streaming.

These tests execute against actual running Kafka broker infrastructure
when explicitly opted into via environment variables.

Execution:
    TEST_LIVE_KAFKA=1 \
    KAFKA_BOOTSTRAP_SERVER="kafka.team.internal:9092" \
    KAFKA_TEST_TOPIC="ml-framework-integration-test-topic" \
    python3 -m unittest -v tests/integration/test_kafka_live.py

Safety & Isolation Guarantees:
- Tests ONLY run when TEST_LIVE_KAFKA=1 is set.
- All test messages include a unique correlation ID per test run.
- Tests use unique, ephemeral consumer groups: ml-framework-live-<uuid>/
- Never alters broker configuration, partitions, or deletes shared application topics.
- All consumer and producer operations have bounded timeouts (max 5.0 seconds).
- Fails clearly if enabled but confluent-kafka or broker connectivity is unavailable.
"""

import os
import sys
import time
import uuid
import unittest
import importlib.util
from pathlib import Path
from typing import List, Dict, Any

# Ensure src/ is on python path
SRC_PATH = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_framework.config.model_config import InputConfig
from ml_framework.input.readers import KafkaTopicReader
from ml_framework.input.processor import InputProcessor
from ml_framework.results.execution_result import ExecutionContext
from ml_framework.errors.exceptions import (
    InvalidInputSourceError,
    ConfigurationError,
)

HAS_CONFLUENT_KAFKA = importlib.util.find_spec("confluent_kafka") is not None


def _load_test_env() -> None:
    """
    Explicitly loads local .env variables into os.environ for integration test execution only.
    Does not overwrite environment variables already set in the shell.
    Uses standard library only (no third-party dependencies).
    """
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_file.is_file():
        return
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip()
                if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
                    v = v[1:-1]
                elif " #" in v:
                    v = v.split(" #", 1)[0].rstrip()
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_test_env()


def is_kafka_enabled() -> bool:
    """Checks whether live Kafka integration testing is explicitly enabled."""
    return os.environ.get("TEST_LIVE_KAFKA", "").strip().lower() in ("1", "true", "yes")


@unittest.skipUnless(is_kafka_enabled(), "Live Kafka testing disabled. Set TEST_LIVE_KAFKA=1 to enable.")
class TestKafkaLiveIntegration(unittest.TestCase):
    """
    Live integration test suite verifying KafkaTopicReader and InputProcessor
    against a live Kafka broker.
    """

    bootstrap_servers = None
    test_topic = None
    kafka_options: Dict[str, Any] = {}
    producer = None

    @classmethod
    def setUpClass(cls):
        # 1. Check client library dependency
        if not HAS_CONFLUENT_KAFKA:
            raise AssertionError(
                "TEST_LIVE_KAFKA is enabled, but 'confluent_kafka' is not installed in the active Python environment. "
                "Install it via: pip install '.[kafka]' or pip install confluent-kafka"
            )

        # 2. Check and validate required configuration
        input_cfg = InputConfig.from_dict(dict(os.environ))
        cls.bootstrap_servers = input_cfg.kafka_bootstrap_servers
        cls.test_topic = input_cfg.kafka_topic
        cls.kafka_options = dict(input_cfg.kafka_options)
        try:
            input_cfg.validate_live_kafka_config()
        except ConfigurationError as cfg_err:
            raise AssertionError(
                f"TEST_LIVE_KAFKA is enabled, but configuration is invalid: {cfg_err}"
            ) from cfg_err

        from confluent_kafka import Producer
        from confluent_kafka.admin import AdminClient, NewTopic

        producer_conf = {
            "bootstrap.servers": cls.bootstrap_servers,
            **cls.kafka_options,
        }

        # 3. Verify connectivity to Kafka broker
        try:
            admin = AdminClient(producer_conf)
            cluster_metadata = admin.list_topics(timeout=5.0)
            if cluster_metadata.topics is None:
                raise RuntimeError("Failed to retrieve Kafka cluster metadata")

            # Check if topic exists; if not, attempt topic creation if permitted
            if cls.test_topic not in cluster_metadata.topics:
                new_topic = NewTopic(cls.test_topic, num_partitions=1, replication_factor=1)
                futures = admin.create_topics([new_topic])
                for topic, future in futures.items():
                    try:
                        future.result(timeout=5.0)
                    except Exception as topic_err:
                        raise AssertionError(
                            f"Live Kafka test topic '{cls.test_topic}' does not exist on '{cls.bootstrap_servers}' "
                            f"and could not be created automatically: {topic_err}. "
                            f"Please create the dedicated topic using: "
                            f"kafka-topics.sh --bootstrap-server {cls.bootstrap_servers} --create --topic {cls.test_topic} --partitions 1 --replication-factor 1"
                        ) from topic_err
        except Exception as conn_err:
            if isinstance(conn_err, AssertionError):
                raise
            raise AssertionError(
                f"Could not connect to live Kafka broker at '{cls.bootstrap_servers}': {conn_err}. "
                f"Verify broker is running, endpoint is reachable, and authentication settings are correct."
            ) from conn_err

        cls.producer = Producer(producer_conf)

    def setUp(self):
        self.correlation_id = str(uuid.uuid4())
        # Use unique consumer group per test to ensure isolated offset tracking
        self.group_id = f"ml-framework-live-group-{uuid.uuid4().hex[:8]}"

    def _publish_message(self, message: str) -> None:
        """Publishes a test message to the dedicated test topic and flushes."""
        self.producer.produce(self.test_topic, message.encode("utf-8"))
        self.producer.flush(timeout=5.0)

    def _publish_messages(self, messages: List[str]) -> None:
        """Publishes multiple test messages and flushes."""
        for msg in messages:
            self.producer.produce(self.test_topic, msg.encode("utf-8"))
        self.producer.flush(timeout=5.0)

    def test_live_topic_validation(self):
        """Verifies topic validation with both valid and invalid topic naming schemes."""
        reader = KafkaTopicReader(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            kafka_options=self.kafka_options,
        )

        # Valid URI scheme and raw topic name
        self.assertTrue(reader.validate(f"kafka://{self.test_topic}"))
        self.assertTrue(reader.validate(self.test_topic))
        self.assertTrue(reader.validate({"kafka_topic": self.test_topic}))

        # Invalid topic name with spaces
        with self.assertRaises(InvalidInputSourceError):
            reader.validate("kafka://invalid topic with spaces")

        # Empty topic name
        with self.assertRaises(InvalidInputSourceError):
            reader.validate("kafka://")

        reader.close()

    def test_live_single_message_consume(self):
        """Verifies publishing and reading a single controlled message from Kafka."""
        payload = f"ml-framework-live-payload-{self.correlation_id}"
        self._publish_message(payload)

        reader = KafkaTopicReader(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            kafka_options=self.kafka_options,
        )

        deadline = time.time() + 10.0
        consumed_messages = []
        while time.time() < deadline:
            for msg in reader.read(f"kafka://{self.test_topic}", batch_size=1, timeout_seconds=2.0):
                consumed_messages.append(msg)
                if payload in msg:
                    break
            if any(payload in m for m in consumed_messages):
                break

        self.assertTrue(
            any(payload in m for m in consumed_messages),
            f"Expected to consume payload containing '{payload}', got {consumed_messages}"
        )
        reader.close()

    def test_live_batch_message_consumption(self):
        """Verifies batch reading when batch_size > 1."""
        batch_payloads = [
            f"batch-msg-0-{self.correlation_id}",
            f"batch-msg-1-{self.correlation_id}",
            f"batch-msg-2-{self.correlation_id}",
        ]
        self._publish_messages(batch_payloads)

        reader = KafkaTopicReader(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            kafka_options=self.kafka_options,
        )

        batches = list(reader.read(
            f"kafka://{self.test_topic}",
            batch_size=3,
            timeout_seconds=5.0
        ))

        self.assertTrue(len(batches) > 0, "Expected at least one batch from Kafka")
        first_batch = batches[0]
        self.assertIsInstance(first_batch, list, "Batch output should be a list when batch_size > 1")
        reader.close()

    def test_live_input_processor_kafka(self):
        """Verifies end-to-end InputProcessor integration with real Kafka."""
        payload = f"processor-kafka-msg-{self.correlation_id}"
        self._publish_message(payload)

        cfg = InputConfig(
            source_type="kafka",
            kafka_bootstrap_servers=self.bootstrap_servers,
            kafka_topic=self.test_topic,
            kafka_group_id=self.group_id,
            kafka_options=self.kafka_options,
        )
        processor = InputProcessor(config=cfg)

        ctx = ExecutionContext()
        result = processor.process(f"kafka://{self.test_topic}", context=ctx)
        self.assertIsNotNone(result)
        self.assertEqual(ctx.metadata.get("input_source_type"), "kafka")
        processor.close()

    def test_live_consumer_cleanup(self):
        """Verifies consumer closure releases sockets and resources."""
        reader = KafkaTopicReader(
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            kafka_options=self.kafka_options,
        )
        # Validate and close
        reader.validate(f"kafka://{self.test_topic}")
        reader.close()
        self.assertIsNone(reader._consumer)


if __name__ == "__main__":
    unittest.main()
