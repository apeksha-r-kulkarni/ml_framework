# Live Integration Testing Guide — MinIO & Kafka Connectors

This guide explains how to configure environment variables, install optional client dependencies, and run live integration tests for `ml_framework` against actual team MinIO and Kafka services.

---

## 1. Overview & Separation Strategy

The ML Framework test suite strictly separates **unit & mocked tests** from **live infrastructure tests**:

| Test Layer | Test Files | Default Execution | Dependencies Required |
| :--- | :--- | :--- | :--- |
| **Unit & Mocked Tests** | `tests/unit/test_*.py`<br>`tests/integration/test_pipeline_integration.py` | **Runs and passes offline** | None (100% mocked / isolated) |
| **Live MinIO Tests** | `tests/integration/test_minio_live.py` | **Skipped** unless opted in | Running MinIO / S3 service + `boto3` |
| **Live Kafka Tests** | `tests/integration/test_kafka_live.py` | **Skipped** unless opted in | Running Kafka broker + `confluent-kafka` |

Running `python3 -m unittest discover tests` will automatically execute all unit and mocked tests and cleanly skip live tests when the opt-in environment variables are not set.

### Opt-in vs Failure Semantics
- **When Opted Out (default):** Both live test suites cleanly skip with a diagnostic message (e.g. `Live MinIO testing disabled. Set TEST_LIVE_MINIO=1 to enable.`).
- **When Opted In (`TEST_LIVE_MINIO=1` or `TEST_LIVE_KAFKA=1`):** Tests are activated. If client libraries (`boto3`, `confluent-kafka`) are missing, configuration is incomplete, or the service is unreachable, tests **fail immediately** with actionable error messages rather than silently skipping. Skipped tests are never reported as verified live tests.

---

## 2. Installing Optional Client Dependencies

The core ML framework remains lightweight with zero external runtime dependencies. Live connectors are configured as optional extras in `pyproject.toml` and `setup.py`:

```bash
# Install MinIO / S3 storage connector (boto3):
pip install -e '.[storage]'

# Install Kafka message streaming connector (confluent-kafka):
pip install -e '.[kafka]'

# Install both live connectors:
pip install -e '.[connectors]'

# Or direct package install:
pip install boto3 confluent-kafka
```

---

## 3. Environment Configuration for Live Testing

The framework connectors read directly from standard `os.environ` variables. The core library (`src/ml_framework`) never auto-loads `.env` files, maintaining clean architectural separation.

For live integration test execution, you can provide configuration using either a local `.env` file or direct shell environment variables.

### Option A: Local `.env` File (Recommended for Local Dev)

A `.env.example` template is provided in the repository root. Copy it to `.env`:

```bash
cp .env.example .env
```

Edit `.env` with your team's authorized service parameters:
- Keep `TEST_LIVE_MINIO=0` and `TEST_LIVE_KAFKA=0` by default.
- Set `TEST_LIVE_MINIO=1` or `TEST_LIVE_KAFKA=1` when you are ready to run tests against active services.
- Replace placeholder values (`<minio-access-key>`, `<kafka-host>`, etc.) with your team's authorized endpoints and credentials.

> [!NOTE]
> `.env` is ignored by Git (`.gitignore`) to prevent accidental commits of local credentials. The live test suites (`tests/integration/test_*_live.py`) load `.env` explicitly during test setup using standard library file parsing without third-party dependencies.

### Option B: Shell Environment Variables (Inline or Export)

You can also provide configuration directly in your shell session without creating a `.env` file:

```bash
# Export MinIO variables in your shell:
export TEST_LIVE_MINIO=1
export MINIO_ENDPOINT="http://minio.team.internal:9000"
export MINIO_ACCESS_KEY="<AUTHORIZED_ACCESS_KEY>"
export MINIO_SECRET_KEY="<AUTHORIZED_SECRET_KEY>"
export MINIO_TEST_BUCKET="ml-framework-integration-test"

# Export Kafka variables in your shell:
export TEST_LIVE_KAFKA=1
export KAFKA_BOOTSTRAP_SERVER="kafka.team.internal:9092"
export KAFKA_TEST_TOPIC="ml-framework-integration-test-topic"
```

> [!CAUTION]
> Never hardcode or commit credentials in version control or in `.env.example`. Always keep secrets in local `.env` or shell variables.

---

## 4. MinIO / S3 Live Integration Testing

### Supported Environment Variables

| Variable | Description | Example |
| :--- | :--- | :--- |
| `TEST_LIVE_MINIO` | **Required Opt-in Flag.** Set to `1` or `true`. | `1` |
| `MINIO_ENDPOINT` *(or `S3_ENDPOINT_URL`)* | HTTP(S) endpoint URL of the team's MinIO service. | `http://minio.team.internal:9000` |
| `MINIO_ACCESS_KEY` *(or `AWS_ACCESS_KEY_ID`)* | Authorized MinIO access key ID. | *Authorized team credential* |
| `MINIO_SECRET_KEY` *(or `AWS_SECRET_ACCESS_KEY`)* | Authorized MinIO secret access key. | *Authorized team credential* |
| `MINIO_TEST_BUCKET` *(or `S3_TEST_BUCKET`)* | Dedicated test bucket (do not use production buckets). | `ml-framework-integration-test` |
| `MINIO_REGION` *(or `AWS_DEFAULT_REGION`)* | Storage region (default: `us-east-1`). | `us-east-1` |
| `MINIO_SECURE` | Set to `true` if endpoint requires TLS/HTTPS without `https://` prefix. | `false` |
| `MINIO_VERIFY_SSL` | Certificate verification setting. Defaults to `true` (strict verification). Set to a custom CA bundle path (e.g. `/etc/ssl/certs/custom-ca.pem`) to trust internal CAs. Set to `false` only in non-production test setups. | `true` |
| `MINIO_CA_BUNDLE` *(or `AWS_CA_BUNDLE`)* | Path to custom CA certificate bundle file for strict TLS verification against internal/private CAs. | `/etc/ssl/certs/team-ca.pem` |
| `MLFLOW_S3_IGNORE_TLS` *(or `MINIO_IGNORE_TLS`)* | Insecure override: set to `true` to bypass TLS verification entirely. Defaults to `false` (verification enabled). Forbidden in production. | `false` |

### MinIO TLS Security Status & Remediation

> [!WARNING]
> **Current Status: BLOCKED**
> Live MinIO integration testing is currently blocked because the internal MinIO endpoint presents an expired, self-signed TLS certificate. In accordance with security policy, TLS certificate verification is **not** disabled.

#### Secure Remediation Options
1. **Preferred (Option 1) — Infrastructure Certificate Renewal:**
   - Infrastructure team re-issues/renews the MinIO service certificate using an authorized corporate/internal CA.
   - The certificate Common Name (CN) and Subject Alternative Names (SAN) must match the MinIO endpoint hostname.
   - The certificate chain must be valid and unexpired.
   - Client applications run with standard defaults (`MINIO_VERIFY_SSL=true`).

2. **Alternative (Option 2) — Approved Corporate CA Bundle:**
   - Infrastructure team supplies the corporate root/intermediate CA certificate bundle.
   - Client runtime specifies the trusted CA bundle path via `MINIO_CA_BUNDLE="/path/to/ca.pem"` (or `AWS_CA_BUNDLE="/path/to/ca.pem"` / `MINIO_VERIFY_SSL="/path/to/ca.pem"`).
   - Boto3 validates the server certificate against this trusted bundle, keeping certificate validation and hostname verification strictly enabled.

#### Follow-Up Verification Procedure (Once Remediated)

Once infrastructure implements Option 1 or Option 2, execute the following 3-step verification suite:

1. **Step 1: MinIO Audio Ingestion via Input Processor:**
   ```bash
   TEST_LIVE_MINIO=1 \
   MINIO_ENDPOINT="https://minio.team.internal:9000" \
   MINIO_ACCESS_KEY="<ACCESS_KEY>" \
   MINIO_SECRET_KEY="<SECRET_KEY>" \
   MINIO_TEST_BUCKET="ml-framework-integration-test" \
   MINIO_CA_BUNDLE="/path/to/ca.pem" \
   python3 -m unittest -v tests/integration/test_minio_live.py
   ```
   Verifies: S3 client connection, bucket access, binary chunked streaming, and `InputProcessor.process()` on MinIO objects.

2. **Step 2: Remote Model Retrieval and Staging via Model Loader:**
   ```bash
   TEST_LIVE_MINIO=1 \
   MINIO_ENDPOINT="https://minio.team.internal:9000" \
   MINIO_ACCESS_KEY="<ACCESS_KEY>" \
   MINIO_SECRET_KEY="<SECRET_KEY>" \
   MINIO_MODEL_BUCKET="speech-data" \
   MINIO_MODEL_KEY="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo" \
   MINIO_CA_BUNDLE="/path/to/ca.pem" \
   python3 -m unittest -v tests/integration/test_multimodel_minio_live.py
   ```
   Verifies: Model artifact downloading, integrity verification, extraction, and staging.

3. **Step 3: End-to-End MinIO-to-Triton Serving:**
   ```bash
   TEST_LIVE_MINIO=1 \
   MINIO_ENDPOINT="https://minio.team.internal:9000" \
   MINIO_ACCESS_KEY="<ACCESS_KEY>" \
   MINIO_SECRET_KEY="<SECRET_KEY>" \
   MINIO_MODEL_BUCKET="speech-data" \
   MINIO_MODEL_KEY="model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo" \
   TRITON_SERVER_URL="http://localhost:8010" \
   MINIO_CA_BUNDLE="/path/to/ca.pem" \
   python3 -m unittest -v tests/integration/test_minio_triton_live.py
   ```
   Verifies: Staging the retrieved model into Triton model repository, probing Triton model readiness, and transcribing test audio through Triton KServe v2 API.

---

## 5. Kafka Live Integration Testing

### Supported Environment Variables

| Variable | Description | Example |
| :--- | :--- | :--- |
| `TEST_LIVE_KAFKA` | **Required Opt-in Flag.** Set to `1` or `true`. | `1` |
| `KAFKA_BOOTSTRAP_SERVER` | Kafka broker host and port. | `kafka.team.internal:9092` |
| `KAFKA_TEST_TOPIC` | Dedicated test topic name. | `ml-framework-integration-test-topic` |
| `KAFKA_GROUP_ID` | Base consumer group identifier. | `ml-framework-live-consumer` |
| `KAFKA_SECURITY_PROTOCOL` | Security protocol (`PLAINTEXT`, `SASL_PLAINTEXT`, `SASL_SSL`, `SSL`). | `PLAINTEXT` |
| `KAFKA_SASL_MECHANISM` | SASL mechanism (`PLAIN`, `SCRAM-SHA-256`, `SCRAM-SHA-512`, `GSSAPI`). | `PLAIN` |
| `KAFKA_SASL_USERNAME` | SASL username (if applicable). | *Authorized credential* |
| `KAFKA_SASL_PASSWORD` | SASL password (if applicable). | *Authorized credential* |
| `KAFKA_SSL_CA_LOCATION` | Path to SSL CA certificate (if applicable). | `/path/to/ca.pem` |
| `KAFKA_SSL_CERTIFICATE_LOCATION` | Path to client certificate (if applicable). | `/path/to/client.pem` |
| `KAFKA_SSL_KEY_LOCATION` | Path to client private key (if applicable). | `/path/to/client.key` |

### Dedicated Test Topic Setup

To prevent consuming or interfering with actual application traffic, use a dedicated integration test topic.

#### Topic Creation via `kafka-topics.sh`
```bash
kafka-topics.sh --bootstrap-server $KAFKA_BOOTSTRAP_SERVER \
  --create --if-not-exists \
  --topic ml-framework-integration-test-topic \
  --partitions 1 \
  --replication-factor 1
```

### Safety & Isolation Guarantees in `test_kafka_live.py`
- **Correlation ID Tracking:** Each test run stamps messages with a unique `uuid4` correlation ID, preventing collisions with concurrent tests.
- **Ephemeral Consumer Groups:** Each test instantiates a fresh consumer group `ml-framework-live-group-<uuid>` with `auto.offset.reset: earliest`, ensuring deterministic offset positioning without altering shared consumer group offsets.
- **Bounded Polling:** Polling loops are bounded by strict timeouts (5.0s max) to fail fast if the broker becomes unreachable.
- **No Topic Deletion:** The suite does not drop or alter broker topic configurations.

### Running Kafka Live Tests
```bash
TEST_LIVE_KAFKA=1 \
KAFKA_BOOTSTRAP_SERVER="kafka.team.internal:9092" \
KAFKA_TEST_TOPIC="ml-framework-integration-test-topic" \
python3 -m unittest -v tests/integration/test_kafka_live.py
```

### Verified Behaviors
- **Topic Name Validation:** Verifies validation of URIs (`kafka://topic-name`), raw names, dictionary configs, and rejection of invalid characters.
- **Single Message Ingestion:** Publishes a message to the broker and consumes it via `KafkaTopicReader.read()`.
- **Batch Message Ingestion:** Publishes multiple messages and reads them in batch slices when `batch_size > 1`.
- **End-to-End Processing:** Verifies that `InputProcessor.process()` resolves and processes messages from Kafka topics into `ExecutionContext`.
- **Consumer Lifecycle Teardown:** Verifies consumer socket release on `.close()`.

---

## 6. Running the Complete Verification Suite

To run all unit tests, pipeline integration tests, and live tests together:

```bash
TEST_LIVE_MINIO=1 \
MINIO_ENDPOINT="http://minio.team.internal:9000" \
MINIO_ACCESS_KEY="<ACCESS_KEY>" \
MINIO_SECRET_KEY="<SECRET_KEY>" \
MINIO_TEST_BUCKET="ml-framework-integration-test" \
TEST_LIVE_KAFKA=1 \
KAFKA_BOOTSTRAP_SERVER="kafka.team.internal:9092" \
KAFKA_TEST_TOPIC="ml-framework-integration-test-topic" \
python3 -m unittest discover tests
```
