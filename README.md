# ML Framework

Standalone, reusable, offline-capable ML execution framework designed for reliable model loading, dynamic inference orchestration, and lifecycle management.

---

## 1. Overview

`ml_framework` provides a unified execution and lifecycle abstraction for machine learning workloads across multiple domains (Text Scraping / NER, Automatic Speech Recognition, Language Identification, and future pipeline models).

The package is strictly decoupled from pipeline ingestion logic, Kafka consumers, Django/React services, and external database backends. Host applications (such as Kafka streaming processors or batch workers) interact with model runtimes through a clean, standardized API:

```python
from ml_framework import ml

# 1. High-level consumer interface: load model with pre-validation
model = ml.load(
    model_source="minio://models/ner.tar.gz",
    input_source="/path/to/input.jsonl",
    framework="transformers",
    device="cpu"
)

# 2. Process data payloads
result = model.process("Apple is looking at buying U.K. startup for $1 billion")

if result.success:
    print("Inference Output:", result.output)
    print("Duration (ms):", result.duration_ms)
else:
    print("Execution Error:", result.error, result.error_type)

# 3. Model lifecycle management
model.unload()  # Frees model weights; sets state to UNINITIALIZED
model.stop()    # Completely terminates the manager and cleans up resources
```

Lower-level access via `ExecutionManager` remains fully supported:

```python
from ml_framework import ExecutionManager, FrameworkConfig, ExecutionContext

config = FrameworkConfig(device="cpu", batch_size=1)
manager = ExecutionManager(config=config)
manager.initialize()
result = manager.execute(input_data="Sample payload")
manager.shutdown()
```

---

## 2. Design Authority and Architecture

The implementation strictly adheres to the six approved architectural diagrams documented in [`docs/architecture.md`](docs/architecture.md):

1. **Successful Execution Sequence:** Explicit 4-stage pipeline (Context validation -> Preprocessing -> Model Inference -> Postprocessing/Formatting -> Telemetry).
2. **Model Loading Sequence:** Cache-first resolution -> MinIO/S3/SeaweedFS retrieval via Boto3 -> Artifact integrity validation -> Format deserialization.
3. **Execution Flow:** High-throughput streaming flow with state checking (`READY`, `PAUSED`, `STOPPED`).
4. **Error Handling Sequence:** Hierarchical error classification (`TransientStorageError`, `ModelFormatError`, `InferenceError`, `InvalidInputSourceError`, `DependencyMissingError`) ensuring host application stability without unhandled process crashes.
5. **Class Diagram:** Modular component architecture with dependency injection for processors, readers, loaders, and status trackers.
6. **Resource Cleanup Sequence:** Explicit memory release, accelerator cache purging, and temporary file clearing on reload/shutdown.

---

## 3. Directory Structure

```text
ml_framework/
├── pyproject.toml              # Modern PEP 517/621 package build configuration
├── setup.py                    # Setuptools build script for offline/editable installs
├── README.md                   # Package documentation and audit status
├── docker/
│   └── triton_nemo/            # Self-contained Triton NeMo Conformer serving container
│       ├── Dockerfile          # Reproducible Triton 23.10 NeMo ASR CPU image
│       ├── requirements.txt    # Pinned NeMo, PyTorch CPU, soundfile dependencies
│       ├── README.md           # Serving instructions & runtime boundary architecture
│       └── models/             # Triton Python backend model repository
├── docs/
│   ├── architecture.md         # Approved sequence diagrams, class diagram, & specs
│   ├── integration_contract.md # Pipeline integration contract & developer guide
│   └── live_integration_testing.md # Live MinIO & Kafka testing guide and TLS remediation
├── src/
│   └── ml_framework/
│       ├── __init__.py         # Public exports (ml, ExecutionManager, Config, Results, Errors)
│       ├── ml.py               # Public API facade (load, validate_input, validate_environment)
│       ├── execution_manager.py# Central lifecycle orchestrator and facade
│       ├── config/             # Typed, validated configuration schemas
│       │   ├── __init__.py
│       │   └── model_config.py # FrameworkConfig, StorageConfig, InputConfig, TritonConfig
│       ├── environment/        # Environment and dependency management
│       │   ├── __init__.py
│       │   └── environment_manager.py # Offline dependency inspector & device validator
│       ├── errors/             # Standardized exception hierarchy
│       │   ├── __init__.py
│       │   └── exceptions.py   # MLFrameworkError and specialized subclasses
│       ├── inference/          # Exclusive Triton production inference execution
│       │   ├── __init__.py
│       │   ├── triton_client.py# Zero-dependency KServe v2 HTTP REST client
│       │   ├── triton_engine.py# TritonInferenceEngine implementing BaseInferenceEngine
│       │   └── tensor_mapper.py# Domain-to-Triton request/response tensor mapper
│       ├── input/              # Input processing and multi-source readers
│       │   ├── __init__.py
│       │   ├── readers.py      # LocalFileReader, ObjectStorageReader, KafkaTopicReader, DirectInputReader
│       │   ├── processor.py    # InputProcessor with pre-load validation & streaming
│       │   └── audio.py        # Audio payload validation, decoding, and resampling
│       ├── interfaces/         # Abstract contracts for component injection
│       │   ├── __init__.py
│       │   └── base.py         # Base loaders, processors, readers, and managers
│       ├── models/             # Model loading, format detection, and model adapters
│       │   ├── __init__.py
│       │   ├── model_loader.py # Generic ModelLoader with remote caching and Triton staging
│       │   ├── specialized_loaders.py # Speech, Spacy, Transformers, ONNX loaders
│       │   ├── speech_adapter.py # HuggingFace & NeMo CTC speech model adapter
│       │   ├── spacy_adapter.py# spaCy NLP pipeline adapter
│       │   ├── transformers_adapter.py # Transformers text model adapter
│       │   ├── onnx_adapter.py # ONNX Runtime model adapter
│       │   └── format_detector.py # Automatic model format and framework detector
│       ├── output/             # Output structuring, normalization, and serialization
│       │   ├── __init__.py
│       │   └── processor.py    # OutputProcessor implementing BaseOutputProcessor
│       ├── results/            # Execution results & telemetry contexts
│       │   ├── __init__.py
│       │   └── execution_result.py # ExecutionResult, ExecutionContext, ExecutionState
│       └── status/             # Observable status tracking and telemetry
│           ├── __init__.py
│           └── tracker.py      # StatusTracker implementing BaseStatusTracker
└── tests/
    ├── __init__.py
    ├── integration/            # Pipeline integration & live infrastructure test suites
    │   ├── __init__.py
    │   ├── test_pipeline_integration.py # 22 offline mock pipeline integration tests
    │   ├── test_triton_live.py # 2 live Triton KServe v2 integration tests
    │   ├── test_kafka_live.py  # 5 live Kafka broker streaming tests
    │   ├── test_minio_live.py  # 5 live MinIO object storage reader tests
    │   ├── test_minio_triton_live.py # 2 live MinIO-to-Triton retrieval & serving tests
    │   └── test_multimodel_minio_live.py # 4 live multi-model MinIO download tests
    ├── manual/                 # Manual test runners and model verification configs
    │   ├── __init__.py
    │   ├── model_test_runner.py# Multi-model verification runner
    │   ├── test_minio_triton_config.py # MinIO & Triton configuration checks
    │   ├── test_runner.py      # Quick test runner script
    │   └── models_to_test.csv  # Verification test model matrix
    └── unit/                   # 17 comprehensive unit test modules (294 automated tests)
        ├── __init__.py
        ├── test_config.py
        ├── test_environment_manager.py
        ├── test_execution_manager.py
        ├── test_input_processing.py
        ├── test_manual_runner.py
        ├── test_minio_to_triton.py
        ├── test_ml_api.py
        ├── test_model_families_config.py
        ├── test_model_loader.py
        ├── test_model_loader_remote.py
        ├── test_model_loader_speech_nemo.py
        ├── test_output_processing.py
        ├── test_package_import.py
        ├── test_speech_inference.py
        ├── test_status_tracking.py
        └── test_triton_inference_engine.py
```

---

## 4. Offline Installation & Air-Gapped Operation

### Zero-Dependency Core
The core foundation of `ml_framework` relies strictly on the **Python standard library** (Python 3.8+). No external packages or internet connections are required to install, import, or run the base orchestration skeleton.

### Offline Installation via Wheels
In air-gapped environments, dependencies can be pre-downloaded to a local directory:

```bash
# 1. On an internet-connected staging machine:
pip download -d /tmp/wheels .

# 2. On the air-gapped host:
pip install --no-index --find-links=/tmp/wheels .

# Or for local editable development:
pip install -e . --no-build-isolation --no-deps
```

### Building the pip Package (Wheel & Source Distribution)

`ml_framework` can be built reproducibly into standard PEP 517/PEP 621 wheels and source distributions using Setuptools:

```bash
# Build both source distribution (sdist) and binary wheel (bdist_wheel)
python3 setup.py sdist bdist_wheel

# Built distributions are output to dist/:
# - dist/ml_framework-0.1.0-py3-none-any.whl
# - dist/ml-framework-0.1.0.tar.gz
```

### Clean Isolated Installation & Verification

Verify installation outside the repository tree without dependencies:

```bash
# 1. Create a clean isolated virtual environment:
python3 -m venv /tmp/isolated_ml_env

# 2. Install the built wheel:
/tmp/isolated_ml_env/bin/pip install dist/ml_framework-0.1.0-py3-none-any.whl

# 3. Test execution completely outside the repository source tree:
cd /tmp && /tmp/isolated_ml_env/bin/python -c "
from ml_framework import ml, ExecutionManager, FrameworkConfig
mgr = ExecutionManager(config=FrameworkConfig(device='cpu'))
mgr.initialize()
res = mgr.execute('Hello world')
print('Execution success:', res.success, '| Output:', res.output)
mgr.shutdown()
"
```

### Optional Domain Extras
Domain-specific dependencies are defined as optional extras in `pyproject.toml` and `setup.py` and are only required when specialized model engines or connectors are loaded:
- `speech`: Audio processing & ASR (`transformers`, `torch`, `soundfile`, `numpy`)
- `transformers`: HuggingFace transformer backends (`torch`, `transformers`)
- `spacy`: spaCy NLP pipelines (`spacy`)
- `onnx`: ONNX Runtime execution (`onnxruntime`, `optimum`)
- `storage`: S3/MinIO/SeaweedFS client SDK (`boto3`)
- `kafka`: Kafka message streaming connector (`confluent-kafka`)
- `connectors`: Both MinIO and Kafka connectors (`boto3`, `confluent-kafka`)
- `dev`: Development and unit testing (`pytest`)
- `all`: Complete set of all optional extras

Install extras using standard pip syntax:
```bash
pip install dist/ml_framework-0.1.0-py3-none-any.whl[speech]
# or during local development:
pip install -e '.[connectors]'
```

---

## 5. Development Tasks Alignment & Audit Evidence

Based on the repository audit conducted on **October 28, 2026**, the implementation and verification status across all eight assigned tasks is documented below:

| Task | Scope | Audit Status | Implementation & Evidence Details |
| :--- | :--- | :--- | :--- |
| **Task 1: Execution Manager Skeleton Development** | Scheduled: Oct 19, 2026 | **PASS** | `ExecutionManager` and public `ml` facade (`ml.load()`, `ml.validate_input()`, `ml.validate_environment()`). Verified with lifecycle state transitions (`UNINITIALIZED` -> `INITIALIZING` -> `READY` -> `STOPPED`), concurrent in-flight execution draining, reload rollback, 28 unit tests, and 22 pipeline integration tests. |
| **Task 2: Input Processing Development** | Scheduled: Oct 19–20, 2026 | **PARTIAL**<br>*(Code: PASS, Infra: BLOCKED)* | Multi-source readers (`LocalFileReader`, `ObjectStorageReader`, `KafkaTopicReader`, `DirectInputReader`) and `InputProcessor`. Local file reading and live Kafka message streaming passed. Remote MinIO live verification is **BLOCKED** by an expired, self-signed TLS certificate (`CN=minio.na.cair.in`, expired March 18, 2026). All 41 offline unit tests passed. |
| **Task 3: Environment Manager Development** | Scheduled: Oct 20–22, 2026 | **PASS** | `EnvironmentManager` offline dependency inspection, device placement resolution (`cpu`, `cuda`, `cuda:<idx>`, `mps`), CPU fallback, accelerator checks, and offline recommendation formatting. Verified with 28 unit tests. |
| **Task 4: Generic Model Loader POC – Speech Model** | Scheduled: Oct 22–26, 2026 | **PARTIAL**<br>*(Code: PASS, Infra: BLOCKED)* | `ModelLoader`, `SpeechModelLoader`, `SpeechModelAdapter`, and `FormatDetector` implemented with offline directory inspection, format detection, and caching. Local model loading and unit tests passed. Remote model artifact download and staging from MinIO is **BLOCKED** due to MinIO's TLS certificate issue. Retains existing NeMo Conformer CTC speech reference implementation. |
| **Task 5: Inference Engine Development** | Scheduled: Oct 26, 2026 | **PASS** | `TritonInferenceEngine` and `TritonClient` implemented using Python standard library `urllib.request` (zero external HTTP dependencies). Delegates production inference exclusively to NVIDIA Triton Inference Server over KServe v2 REST protocol. Verified with 2 live Triton integration tests against running container. |
| **Task 6: Output Processing Development** | Scheduled: Oct 26–27, 2026 | **PASS** | `OutputProcessor` implementing `BaseOutputProcessor` with task-specific normalization (transcription, language detection, denoising), confidence filtering, and JSON-safe primitive serialization (`to_dict`, `to_json`). Verified with unit tests. Calling applications handle downstream publishing (Kafka, MinIO, DB, UI). |
| **Task 7: Execution Manager and Status Tracking Development** | Scheduled: Oct 27–29, 2026 | **PASS** | `StatusTracker` implementing `BaseStatusTracker` with thread-safe counters, per-stage latency tracking (`duration_ms`), success/failure rates, last error recording, and lifecycle observability. Integrated with `ExecutionManager` state transitions. Verified with unit tests. |
| **Task 8: Error Handling and Resource Cleanup Development** | Scheduled: Oct 29–30, 2026 | **PASS** | Exception taxonomy under `MLFrameworkError`, `BaseResourceManager` cleanup, in-flight execution draining, model weight unloading, rollback on reload/staging failure, and connection pool cleanup across readers (`KafkaTopicReader.close()`, `ObjectStorageReader.close()`, `ModelLoader.close()`). Verified with comprehensive unit and integration tests. |

### Test Suite Execution & Evidence Distinction

To clearly distinguish automated test results from manual live tests:

| Test Layer | Test Count | Environment | Result | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Automated Unit Tests** | 296 tests across 16 files | Offline / Mocked | **PASS** | Automated suite testing configuration, environment checks, execution lifecycle, input reading, audio decoding, model loaders, and status tracking. |
| **Pipeline Integration Tests** | 22 tests (`test_pipeline_integration.py`) | Offline / Mocked | **PASS** | Automated suite testing external pipeline integration contracts, stage timeouts, and error handling. |
| **Live Triton Integration Tests** | 2 tests (`test_triton_live.py`) | Running Triton Container | **PASS** (When container active) | Manual live test verifying server/model readiness and audio transcription (`test3.wav`) via `ml.load()`. Skips cleanly when container inactive. |
| **Live Kafka Streaming Tests** | 5 tests (`test_kafka_live.py`) | Running Kafka Broker | **PASS (Live)** | Verified live broker polling when `confluent-kafka` dependency staging was supplied. |
| **Live MinIO Storage Tests** | Tests in `test_minio_live.py`, `test_minio_triton_live.py` | Remote MinIO Service | **BLOCKED** | Blocked by MinIO's expired, self-signed TLS certificate (`CN=minio.na.cair.in`, expired March 18, 2026 GMT). TLS verification is strictly maintained. |

### Documented Limitations & Operational Boundaries

1. **MinIO End-to-End Verification is Blocked:**
   - The remote MinIO service (`https://192.168.2.175:9000`) presents an expired, self-signed TLS certificate (`CN=minio.na.cair.in`, expired March 18, 2026 GMT).
   - Strict TLS certificate verification is enforced; bypassing TLS (`verify_ssl=False`) is prohibited by security policy.
   - Full live verification of Tasks 2 and 4 requires infrastructure remediation (renewed certificate or approved corporate CA bundle). See [`docs/live_integration_testing.md`](docs/live_integration_testing.md) for details.

2. **Kafka Execution Runtime:**
   - The Kafka consumer is executed in the host application runtime (e.g. streaming ingestion worker), NOT inside the NVIDIA Triton container.
   - `confluent-kafka` is declared as an optional connector extra (`ml-framework[kafka]`).
   - Broker port 9095 is reachable. Live Kafka integration tests require setting `TEST_LIVE_KAFKA=1`.

3. **Downstream Publishing is Outside Framework Scope:**
   - `ml_framework` returns execution results (`ExecutionResult`) directly to its caller.
   - Downstream publishing (e.g. producing messages to Kafka output topics, storing records in databases, or pushing to webhooks) is the responsibility of the host application, not the framework.

4. **Task Verification Scope:**
   - Tasks 1, 3, 5, 6, 7, and 8 are fully verified (**PASS**).
   - Tasks 2 and 4 are complete in code and pass all unit tests, but remain **PARTIAL** in live environments pending MinIO TLS certificate remediation by the infrastructure team.

---

## 6. Live Infrastructure Integration Testing

For testing against real team MinIO and Kafka services:
- **MinIO Live Suite:** `tests/integration/test_minio_live.py` (enabled via `TEST_LIVE_MINIO=1`)
- **MinIO-to-Triton Live Suite:** `tests/integration/test_minio_triton_live.py` (enabled via `TEST_LIVE_MINIO=1`)
- **Kafka Live Suite:** `tests/integration/test_kafka_live.py` (enabled via `TEST_LIVE_KAFKA=1`)
- **Triton Live Suite:** `tests/integration/test_triton_live.py` (enabled when Triton container is active)
- **Configuration & Remediation Guide:** [`docs/live_integration_testing.md`](docs/live_integration_testing.md)
