# ML Framework — External Pipeline Integration Contract

**Document Version:** 1.0.0  
**Target Audience:** Independent Pipeline Developers (e.g., Text Pipeline, ASR Workers, LID Consumers)  
**Package:** `ml_framework`  
**Programming Language:** Python 3.9+ (Standard Library Foundation)  

---

## 1. Overview & Architecture

The `ml_framework` package provides a standalone, reusable, air-gapped-compatible Python runtime for orchestrating model loading, dynamic inference, and execution lifecycle management.

### Architectural Boundary
* **Pure Python Library:** The framework is consumed as an imported Python library (`from ml_framework import ml`). It does **not** expose or require an HTTP server, REST endpoints, microservice daemon, or external database.
* **Pipeline Decoupling:** Consumer pipelines (such as the Text Scraping / NER Pipeline) control their own ingestion loops, preprocessing business logic, and downstream persistence, while delegating model lifecycle and inference orchestration to `ml_framework`.
* **Safe Crash Isolation:** Pipeline stage failures and timeouts are trapped centrally and returned inside a structured [`ExecutionResult`](#4-executionresult-contract), ensuring consumer host processes never terminate unexpectedly.

---

## 2. Importing and Initializing the Framework

### Standard Entry Point: The `ml` Facade
The recommended consumer workflow uses the high-level `ml` facade:

```python
from ml_framework import ml, FrameworkConfig, ExecutionContext

# 1. Initialize and load model with automatic pre-validation
model = ml.load(
    model_source="minio://models/ner_roberta.tar.gz",
    framework="transformers",
    device="cpu",
    batch_size=1,
    timeout_seconds=30.0,
)

# 2. Verify readiness
assert model.is_ready
```

### Alternative: Lower-Level `ExecutionManager`
Host applications requiring manual lifecycle wiring or customized dependency injection may instantiate [`ExecutionManager`](file:///home/apeksha-ssi021/Shyena/ml_framework/src/ml_framework/execution_manager.py) directly:

```python
from ml_framework import ExecutionManager, FrameworkConfig

config = FrameworkConfig(
    model_name="ner_pipeline",
    model_version="1.0.0",
    model_source="/models/ner_v1",
    device="cpu",
    timeout_seconds=15.0,
)

manager = ExecutionManager(config=config)
manager.initialize()  # Pre-validates input and environment, transitions to READY
```

---

## 3. Configuration & Input Supply

### Configuration Schemas

#### [`FrameworkConfig`](file:///home/apeksha-ssi021/Shyena/ml_framework/src/ml_framework/config/model_config.py#L115-L197)
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `Optional[str]` | `None` | Logical identifier of the model. |
| `model_version` | `Optional[str]` | `None` | Semantic version string of the model. |
| `model_source` | `Optional[str]` | `None` | Local directory path or object storage URI (`s3://`, `minio://`, `seaweedfs://`). |
| `framework` | `str` | `"auto"` | Target ML framework (`"transformers"`, `"spacy"`, `"onnx"`, `"speech"`). |
| `deployment_point`| `Optional[str]` | `None` | Deployment tag or routing alias. |
| `device` | `str` | `"auto"` | Compute target: `"auto"`, `"cpu"`, `"cuda"`, or `"mps"`. |
| `batch_size` | `int` | `1` | Batch size for inference execution ($\ge 1$). |
| `timeout_seconds` | `float` | `30.0` | Maximum allowed execution time in seconds ($> 0.0$). |
| `enable_telemetry`| `bool` | `True` | Whether to track per-stage execution durations. |
| `auto_cleanup` | `bool` | `True` | Whether to trigger resource cleanup on failure. |
| `storage_config` | `StorageConfig` | `StorageConfig()` | S3/MinIO/SeaweedFS credentials and endpoint settings. |
| `input_config` | `InputConfig` | `InputConfig()` | Input ingestion, batching, and reader settings. |
| `triton_config` | `TritonConfig` | `TritonConfig()` | NVIDIA Triton Inference Server execution and tensor settings. |
| `input_source` | `Optional[Any]` | `None` | Optional pre-load input source to pre-validate. |

#### Storage Configuration (`StorageConfig`)
Configures object storage access when models or data are stored in SeaweedFS, MinIO, or S3:
```python
from ml_framework import StorageConfig

storage_cfg = StorageConfig(
    endpoint_url="http://minio:9000",
    access_key="minioadmin",
    secret_key="miniopassword",
    region="us-east-1",
    secure=False,
)
```

#### Triton Inference Configuration (`TritonConfig`)
Configures exclusive production model serving via NVIDIA Triton Inference Server (Option B: direct NeMo serving via Triton Python backend):
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `endpoint_url` | `str` | `"http://localhost:8010"` | Triton HTTP KServe v2 endpoint URL. |
| `protocol` | `str` | `"http"` | Communication protocol (`"http"` or `"grpc"`). |
| `model_name` | `Optional[str]` | `None` | Name of the model deployed on Triton (e.g., `"nemo_conformer_python"`). |
| `model_version` | `str` | `""` | Version of the model (empty string for latest). |
| `timeout_seconds` | `float` | `30.0` | Maximum timeout for inference execution ($> 0$). |
| `check_model_ready` | `bool` | `True` | Probes server health and model readiness during initialization. |
| `max_retries` | `int` | `3` | Maximum retry attempts on transient network failures. |
| `input_tensor_map` | `dict` | `{}` | Custom input tensor name mapping. |
| `output_tensor_map` | `dict` | `{}` | Custom output tensor name mapping. |

```python
from ml_framework import TritonConfig

triton_cfg = TritonConfig(
    endpoint_url="http://localhost:8010",
    model_name="nemo_conformer_python",
    check_model_ready=True,
    timeout_seconds=30.0,
)
```

### Supported Input Modalities
The framework accepts multiple input modalities via `model.process(input_data)`:

1. **Direct In-Memory Text:**
   ```python
   result = model.process("Sample raw text article to be parsed for named entities.")
   ```
2. **Direct In-Memory Collection / Dictionary:**
   ```python
   result = model.process(["Text item 1", "Text item 2"])
   ```
3. **Local Filesystem Path:**
   ```python
   result = model.process("/path/to/documents.jsonl")
   result = model.process("file:///path/to/documents.txt")
   ```
4. **Remote Object Storage (SeaweedFS / MinIO / S3):**
   ```python
   result = model.process("minio://raw-data/2026/article_01.json")
   result = model.process("seaweedfs://bucket/data.txt")
   ```
5. **Streaming Kafka Topic Reference:**
   ```python
   result = model.process("kafka://raw-text-stream")
   ```

---

## 4. Invoking Processing & `ExecutionResult` Contract

### Processing Invocation
```python
# Pass optional execution context for tracing
ctx = ExecutionContext(
    correlation_id="pipeline-req-8841",
    metadata={"pipeline": "text_ner", "batch_index": 42}
)

result = model.process(input_data="Apple acquired a London AI startup.", context=ctx)
```

### Structured Output: [`ExecutionResult`](file:///home/apeksha-ssi021/Shyena/ml_framework/src/ml_framework/results/execution_result.py#L54-L82)
Every execution returns an `ExecutionResult` object with a guaranteed schema:

```python
@dataclass
class ExecutionResult:
    success: bool                          # True if all stages completed successfully
    output: Any = None                     # Final output payload (or None on failure)
    duration_ms: float = 0.0               # Total end-to-end execution duration in ms
    context: ExecutionContext = ...        # Context containing correlation ID and stage timings
    state: ExecutionState = READY          # Final execution state (READY or FAILED)
    error: Optional[str] = None            # Human-readable error message on failure
    error_type: Optional[str] = None       # Class name of exception (e.g. 'InferenceError')
    applied_model_id: Optional[str] = None # Model ID used for this execution
```

### Checking Success and Retrieving Output
```python
if result.success:
    # 1. Retrieve output payload
    entities = result.output
    print(f"Inference output: {entities}")
    print(f"Total latency: {result.duration_ms} ms")
    
    # 2. Inspect stage latencies
    print("Stage durations:", result.context.stage_durations)
    # e.g.: {"input_processing": 1.2, "inference": 14.5, "output_processing": 0.8}
else:
    # Handle failure gracefully without host process crash
    print(f"Processing failed: [{result.error_type}] {result.error}")
```

### Serialization Contract
To forward results across network boundaries or write to log sinks:
```python
result_dict = result.to_dict()
# {
#     "success": True,
#     "output": [...],
#     "duration_ms": 16.5,
#     "applied_model_id": "ner_pipeline:1.0.0",
#     "state": "READY",
#     "error": None,
#     "error_type": None,
#     "context": {
#         "correlation_id": "pipeline-req-8841",
#         "metadata": {"pipeline": "text_ner", "batch_index": 42},
#         "created_at": 1776680400.0,
#         "stage_durations": {"input_processing": 1.2, "inference": 14.5, "output_processing": 0.8}
#     }
# }
```

---

## 5. Execution Context and Correlation Metadata

The pipeline should supply an [`ExecutionContext`](file:///home/apeksha-ssi021/Shyena/ml_framework/src/ml_framework/results/execution_result.py#L30-L51) to correlate log lines and measure individual pipeline stage timings:

```python
from ml_framework import ExecutionContext

# Context with explicit correlation ID and custom tracing metadata
context = ExecutionContext(
    correlation_id="kafka-msg-offset-102938",
    metadata={
        "source_topic": "raw_scraped_articles",
        "document_type": "news",
        "consumer_group": "ner_worker_v1"
    }
)

result = model.process("Sample payload", context=context)
assert result.context.correlation_id == "kafka-msg-offset-102938"
```

If no `context` is provided, the framework automatically generates one with a UUID4 correlation ID.

---

## 6. Error Handling & Failure Scenarios

The framework strictly distinguishes **synchronous pre-validation errors** from **runtime execution errors**:

```text
               Caller Invocation
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
Initialization / Config        Runtime execute() / process()
(Synchronous Exceptions)       (Encapsulated in ExecutionResult)
       │                             │
       ├─ ConfigurationError         ├─ result.success == False
       ├─ InvalidInputSourceError    ├─ result.error == str(exc)
       ├─ DependencyMissingError     ├─ result.error_type == 'InferenceError'
       ├─ HardwareIncompatibleError  └─ result.state == ExecutionState.FAILED
       └─ ModelLoadError
```

### 1. Pre-Load Validation Errors
Raised immediately as typed exceptions during `ml.load(...)`, `ml.validate_input(...)`, or `manager.initialize()`:
* **`ConfigurationError`:** Raised when configuration parameters violate constraints (e.g., `batch_size < 1`, `timeout_seconds <= 0`, unsupported `device`).
* **`InvalidInputSourceError`:** Raised when an `input_source` path does not exist, an object storage key is missing, or a URI scheme is unsupported.
* **`DependencyMissingError`:** Raised during offline environment check if required Python packages are missing. Details include an actionable offline `pip install` command.
* **`HardwareIncompatibleError`:** Raised if requested hardware accelerator (`"cuda"` or `"mps"`) is not available.

### 2. Runtime Pipeline Failures
Caught internally by `ExecutionManager.execute()` and returned as `ExecutionResult(success=False, ...)`:
* **`InferenceError`:** Model forward pass failure.
* **`InputProcessingError`:** Input data corrupted or unreadable at runtime.
* **`OutputProcessingError`:** Postprocessing or formatting failure.

### 3. Timeout Handling
If pipeline execution exceeds `config.timeout_seconds`:
* Execution returns `ExecutionResult(success=False, error_type="ExecutionTimeoutError")`.
* `result.state` is set to `ExecutionState.FAILED`.
* Failure metrics are incremented strictly once.
* Late worker thread completion will **never** trigger duplicate callbacks or duplicate metric updates.

### 4. Operational Pause State
If the manager is paused via `model.pause()`, calling `model.process()` returns:
* `ExecutionResult(success=False, error_type="StateError", error="ExecutionManager is currently PAUSED...")`.
* Consumers can check this to pause message consumption from queues until `model.resume()` is called.

---

## 7. Lifecycle & Resource Teardown

Consumer pipelines must properly release model memory and execution threads when stopping workers or updating models:

```python
# Option 1: Unload active model weights (frees model memory; state becomes UNINITIALIZED)
model.unload()

# Option 2: Full teardown (shuts down thread pool, closes readers/connectors; state becomes STOPPED)
model.stop()
```

### In-Flight Drain Protection
When `unload()`, `reload()`, or `stop()` is called:
* The framework waits for active in-flight executions to drain (up to `timeout_seconds`).
* New execution requests entering during teardown are safely rejected with `StateError`.
* If draining times out, an exception is raised and the active model is preserved to avoid state corruption.

---

## 8. Supported Behavior vs. Planned Functionality

| Capability | Current Status in `ml_framework` | Consumer Guidance for Pipelines |
| :--- | :--- | :--- |
| **Public Python API (`ml.load`, `process`, `unload`, `stop`)** | **Supported & Verified** | Consume directly via `from ml_framework import ml`. |
| **Typed Configuration (`FrameworkConfig`, `InputConfig`, `TritonConfig`)** | **Supported & Verified** | Pass configuration dataclasses or dictionaries. |
| **Structured Results (`ExecutionResult`, `ExecutionContext`)** | **Supported & Verified** | Rely on `result.success`, `result.output`, `result.context.stage_durations`. |
| **Multi-Source Input Readers (Local, S3/MinIO, Kafka, Direct)** | **Supported & Verified** | Pass path strings, storage URIs, or direct strings. Live integration suites available for MinIO and Kafka (see [`docs/live_integration_testing.md`](live_integration_testing.md)). |
| **Pre-load Source Validation (`validate_source`)** | **Supported & Verified** | Fails fast before model loading on bad sources. |
| **Offline Environment Checks (`EnvironmentManager`)** | **Supported & Verified** | Verifies packages and GPU placement without network access. |
| **Timeout & Concurrency Control** | **Supported & Verified** | Configure `timeout_seconds` on `FrameworkConfig`. |
| **Live MinIO & Kafka Integration Suites** | **Supported & Verified** | Opt-in live test suites in `tests/integration/test_minio_live.py` and `test_kafka_live.py`. |
| **Generic Model Loader (Speech / Transformers / spaCy / ONNX)** | **Supported & Verified** | `ModelLoader` handles local and remote (MinIO/S3) staging, unpacking, and adapter resolution. |
| **Exclusive Production Inference Engine (NVIDIA Triton)** | **Supported & Verified** | `TritonInferenceEngine` delegates production inference to NVIDIA Triton Server. Direct adapter prediction is disabled. |
| **Speech Model Serving (Option B: NeMo Triton Python Backend)** | **Supported & Verified** | Native `.nemo` serving via Triton Python backend (`nemo_conformer_python`), completely independent of ONNX. |
| **Domain Output Processors** | *Planned (Task 6)* | Framework currently defaults to passthrough / normalization of raw inference output. |
| **Database Execution Record Persistence** | *Planned (Task 7)* | Telemetry is currently in-memory via `result.context` and callbacks. |
| **System Resource Manager (GPU Cache Purge, Disk Purge)** | *Planned (Task 8)* | Cleanups trigger injected resource managers; standard GC runs automatically. |

---

## 9. Speech Model Integration (Option B: Triton NeMo Serving)

To run speech-to-text inference with the NeMo Conformer CTC model served via Triton's Python backend:

```python
from ml_framework import ml, FrameworkConfig, TritonConfig

# 1. Configure framework with Triton Option B settings
config = FrameworkConfig(
    model_name="nemo_conformer_python",
    timeout_seconds=60.0,
    triton_config=TritonConfig(
        endpoint_url="http://localhost:8010",
        model_name="nemo_conformer_python",
        check_model_ready=True,
    ),
)

# 2. Initialize runtime (validates Triton liveness and model readiness)
model = ml.load(config=config)

# 3. Process audio input (file path, WAV bytes, or NumPy waveform)
result = model.process("/path/to/audio.wav")
if result.success:
    print("Transcription:", result.output["transcription"])
    print("Duration (ms):", result.duration_ms)
else:
    print("Inference error:", result.error)

# 4. Teardown when pipeline stops
model.stop()
```

---

## 10. Integration Checklist for Independent Pipelines

1. [ ] Install `ml-framework` in editable or wheel mode: `pip install -e /path/to/ml_framework --no-deps`.
2. [ ] Import `ml` and configuration from `ml_framework`.
3. [ ] Configure `FrameworkConfig(device="cpu", timeout_seconds=30.0)`.
4. [ ] Initialize model runtime via `model = ml.load(config=config, ...)`.
5. [ ] Pass `ExecutionContext(correlation_id=...)` with incoming requests for end-to-end tracing.
6. [ ] Inspect `result.success` before consuming `result.output`.
7. [ ] Ensure worker termination triggers `model.stop()` in signal handlers (`SIGTERM`, `SIGINT`).
