# ML Framework — Architectural Specification & Sequence Reference

This document formalizes the architecture of the `ml_framework` package based on the six approved design authority diagrams:
1. **Successful Execution Sequence**
2. **Model Loading Sequence**
3. **Execution Flow**
4. **Error Handling Sequence**
5. **Class Diagram**
6. **Resource Cleanup Sequence**

---

## 1. Successful Execution Sequence

Illustrates the end-to-end execution flow when an input payload is processed successfully without errors.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Consumer (e.g. Text Pipeline)
    participant EM as ExecutionManager
    participant Env as EnvironmentManager
    participant ML as ModelLoader
    participant IP as InputProcessor
    participant IE as InferenceEngine
    participant OP as OutputProcessor
    participant ST as StatusTracker

    Client->>EM: initialize()
    EM->>Env: validate_environment(config)
    Env-->>EM: Environment OK
    EM->>ML: load(model_source)
    ML-->>EM: Model Instance Ready
    EM-->>Client: Ready (State: READY)

    Client->>EM: execute(input_data, context)
    EM->>EM: Create/Validate ExecutionContext
    EM->>IP: process(input_data, context)
    IP-->>EM: preprocessed_input
    EM->>IE: infer(model, preprocessed_input, context)
    IE-->>EM: raw_output
    EM->>OP: process(raw_output, context)
    OP-->>EM: final_output
    EM->>ST: record_success(result)
    ST-->>EM: Metrics updated
    EM-->>Client: ExecutionResult(success=True, output=final_output)
```

---

## 2. Model Loading Sequence

Illustrates dynamic model resolution, artifact downloading (via Boto3 for S3/MinIO), format detection, and adapter instantiation.

```mermaid
sequenceDiagram
    autonumber
    participant EM as ExecutionManager
    participant ML as ModelLoader
    participant SD as S3ArtifactDownloader (Boto3)
    participant Det as FormatDetector
    participant Adap as FrameworkAdapter (spaCy/HF/ONNX/Speech)

    EM->>ML: load(model_source, config)
    alt Remote S3/MinIO Source
        ML->>SD: download_artifact(s3_uri, target_dir)
        SD-->>ML: local_dir_path
    else Local Directory
        ML->>ML: Verify local path exists
    end
    ML->>Det: detect_format(model_dir)
    Det-->>ML: (framework, architecture, model_type)
    ML->>Adap: instantiate_runner(model_dir)
    Adap->>Adap: load_model_weights()
    Adap-->>ML: runner_instance
    ML-->>EM: active_model_instance
```

---

## 3. Execution Flow

Illustrates the internal state transitions and latency tracking across lifecycle stages inside `ExecutionManager`.

```mermaid
stateDiagram-v2
    [*] --> UNINITIALIZED
    UNINITIALIZED --> INITIALIZING: initialize()
    INITIALIZING --> READY: Setup Successful
    INITIALIZING --> FAILED: Dependency / Model Error
    
    READY --> EXECUTING: execute(input)
    state EXECUTING {
        [*] --> InputProcessing
        InputProcessing --> Inference: Preprocessed OK
        Inference --> OutputProcessing: Raw Output OK
        OutputProcessing --> TelemetryRecording: Final Result OK
        TelemetryRecording --> [*]
    }
    EXECUTING --> READY: Success
    EXECUTING --> FAILED: Stage Error Trapped
    
    READY --> PAUSED: pause()
    PAUSED --> READY: resume()
    
    READY --> STOPPED: shutdown()
    FAILED --> STOPPED: shutdown()
    STOPPED --> [*]
```

---

## 4. Error Handling Sequence

Illustrates centralized error trapping, typed classification, and safe caller notification.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Consumer
    participant EM as ExecutionManager
    participant Stage as Pipeline Stage (Input/Inference/Output)
    participant ST as StatusTracker
    participant RM as ResourceManager

    Client->>EM: execute(input_data)
    EM->>Stage: invoke stage
    Stage-->>EM: Exception (e.g. InferenceError)
    EM->>EM: Trap exception & classify error type
    EM->>EM: Measure duration & record failure metrics
    EM->>ST: record_failure(exception, context)
    ST-->>EM: Telemetry updated
    alt Auto-cleanup on failure enabled
        EM->>RM: cleanup_transient_resources()
    end
    EM-->>Client: ExecutionResult(success=False, error=str(e), state=FAILED)
```

---

## 5. Class Diagram

Defines the core class relationships, configuration structures, and interfaces.

```mermaid
classDiagram
    class ExecutionManager {
        -FrameworkConfig config
        -ExecutionState state
        -BaseModelLoader model_loader
        -BaseInputProcessor input_processor
        -BaseInferenceEngine inference_engine
        -BaseOutputProcessor output_processor
        -BaseStatusTracker status_tracker
        -BaseResourceManager resource_manager
        +initialize() bool
        +execute(input_data, context) ExecutionResult
        +pause() void
        +resume() void
        +reload(new_source) bool
        +shutdown() void
        +register_status_callback(callback) void
    }

    class FrameworkConfig {
        +str model_name
        +str model_version
        +str model_source
        +str framework
        +str deployment_point
        +str device
        +int batch_size
        +float timeout_seconds
        +bool enable_telemetry
        +StorageConfig storage_config
        +TritonConfig triton_config
        +validate() void
        +from_dict(data) FrameworkConfig
    }

    class StorageConfig {
        +str endpoint_url
        +str access_key
        +str secret_key
        +str region
        +bool secure
        +from_dict(data) StorageConfig
    }

    class TritonConfig {
        +str endpoint_url
        +str protocol
        +str model_name
        +str model_version
        +float timeout_seconds
        +bool check_model_ready
        +int max_retries
        +dict input_tensor_map
        +dict output_tensor_map
        +validate() void
        +from_dict(data) TritonConfig
    }

    class TritonInferenceEngine {
        -TritonConfig config
        -TritonClient client
        -TensorMapper tensor_mapper
        +verify_readiness(model_name, model_version) bool
        +resolve_model_name(model, context) str
        +infer(model, preprocessed_input, context) Any
    }

    class BaseInferenceEngine {
        <<Interface>>
        +infer(model, preprocessed_input, context) Any
    }

    class ExecutionContext {
        +str correlation_id
        +dict metadata
        +float created_at
        +dict stage_durations
        +record_stage(name, duration_ms) void
    }

    class ExecutionResult {
        +bool success
        +Any output
        +float duration_ms
        +ExecutionContext context
        +ExecutionState state
        +str error
        +str applied_model_id
        +is_success() bool
        +to_dict() dict
    }

    class MLFrameworkError {
        <<Exception>>
        +str message
        +dict details
    }

    ExecutionManager --> FrameworkConfig : configured by
    FrameworkConfig --> StorageConfig : contains
    FrameworkConfig --> TritonConfig : contains
    TritonInferenceEngine ..|> BaseInferenceEngine : implements
    ExecutionManager --> BaseInferenceEngine : delegates inference
    ExecutionManager ..> ExecutionResult : produces
    ExecutionResult --> ExecutionContext : contains
    ExecutionManager ..> MLFrameworkError : handles
```

---

## 6. Resource Cleanup Sequence

Illustrates resource deallocation, memory release, and temporary artifact purge during model switching or shutdown.

```mermaid
sequenceDiagram
    autonumber
    actor Client as Consumer / Lifecycle
    participant EM as ExecutionManager
    participant RM as ResourceManager
    participant ML as ModelLoader
    participant GC as System GC & CUDA

    Client->>EM: shutdown() / reload()
    EM->>RM: cleanup()
    RM->>ML: unload(active_model)
    ML->>ML: Delete local model weights
    RM->>GC: torch.cuda.empty_cache() (if CUDA)
    RM->>GC: gc.collect()
    RM->>RM: Remove temporary download dirs
    RM-->>EM: Cleanup Completed
    EM->>EM: State = STOPPED
    EM-->>Client: Shutdown Confirmed
```

---

## 7. Offline Installation & Dependency Boundaries

To support secure air-gapped or restricted offline environments:

1. **Zero Base Dependencies:**
   - The core package foundation (`ml_framework.execution_manager`, `config`, `results`, `errors`, `interfaces`) relies strictly on standard library Python modules (`dataclasses`, `enum`, `typing`, `time`, `uuid`, `logging`, `abc`).
2. **Offline Installation:**
   - Standard editable install: `pip install --no-index --find-links=/path/to/local_wheels -e /home/apeksha-ssi021/Shyena/ml_framework`
   - Build wheel offline: `python3 -m pip wheel --no-deps /home/apeksha-ssi021/Shyena/ml_framework -w ./dist`
3. **Optional Framework Extras:**
   - Heavy dependencies are segregated into `extras_require`:
     * `ml-framework[storage]`: `boto3`
     * `ml-framework[spacy]`: `spacy`
     * `ml-framework[transformers]`: `transformers`, `torch`
     * `ml-framework[onnx]`: `onnxruntime`, `optimum`
     * `ml-framework[speech]`: `soundfile`, `transformers`, `torch`

---

## 8. Public API Facade (`from ml_framework import ml`)

To provide an intuitive, high-level developer experience without breaking lower-level contracts:

```python
from ml_framework import ml

# 1. Load model with automatic pre-validation and environment checks
model = ml.load(
    model_source="minio://models/ner.tar.gz",
    input_source="/path/to/documents.jsonl",
    framework="transformers",
    device="cpu"
)

# 2. Process inputs dynamically
result = model.process(input_data="Sample raw text payload")
if result.success:
    print("Inference output:", result.output)
else:
    print("Execution failed:", result.error, result.error_type)

# 3. Model lifecycle management
model.unload()  # Frees model memory; transitions state to UNINITIALIZED
model.stop()    # Completely shuts down the manager and releases all resources
```

---

## 9. Input Processing Subsystem (Task 1)

The input subsystem abstracts data ingestion behind the `BaseInputReader` contract, supporting multi-source reading, early configuration validation, and chunked streaming:

```text
                  +-------------------------+
                  |     InputProcessor      |
                  +-------------------------+
                               |
            +------------------+------------------+
            |                  |                  |
+--------------------+ +--------------------+ +--------------------+
|   LocalFileReader  | | ObjectStorageReader| |  KafkaTopicReader  |
| (file://, /path)   | |(SeaweedFS/MinIO/S3)| |  (kafka://topic)   |
| - Chunked stream   | | - Boto3 S3 stream  | | - Batch polling    |
| - Line generators  | | - iter_chunks()    | | - Auto offset      |
+--------------------+ +--------------------+ +--------------------+
```

### Key Capabilities:
- **Pre-Loading Validation:** Validates that input paths exist, object storage keys are present, and Kafka topic names are valid *before* triggering expensive model downloading and instantiation. If invalid, the manager fails early and cleans up gracefully.
- **Memory Safety via Chunking:** Large files and remote streams are read in configurable byte chunks (`chunk_size`) or line batches (`batch_size`) using Python generators. Multi-gigabyte files are never buffered into RAM all at once.
- **Object Storage Support (SeaweedFS / MinIO):** Implemented via standard S3 protocol using Boto3, parsing `seaweedfs://`, `minio://`, and `s3://` URIs.
- **Mockable Architecture:** All readers support dependency injection of mock clients (mock S3, mock Kafka consumer) for unit testing without live services.

---

## 10. Environment Management Subsystem (Task 2)

The `EnvironmentManager` enforces runtime dependency and hardware device integrity in strictly offline environments:

### Key Capabilities:
- **Model Metadata Inspection:** Inspects model directories (`config.json`, `meta.json`, `config.cfg`, `.onnx` presence) and framework configurations to automatically determine exact library requirements (`torch`, `transformers`, `spacy`, `onnxruntime`, `soundfile`, `boto3`, `confluent_kafka`).
- **Offline Dependency Verification:** Utilizes `importlib.util.find_spec` to verify installed packages locally. Does **not** download packages, invoke `pip`, or query external networks.
- **Clear Error Reporting:** When dependencies are missing, raises `DependencyMissingError` with exact missing package names and a ready-to-run offline installation command (`pip install --no-index --find-links=/path/to/wheels ...`).
- **Hardware Compatibility Checking:** Validates device placement (`cpu`, `cuda`, `mps`). Queries CUDA/MPS availability safely and rejects incompatible device requests early with `HardwareIncompatibleError`.

---

## 11. Exclusive Inference Engine: NVIDIA Triton Inference Server (Option B)

In production environments, `ml_framework` enforces **NVIDIA Triton Inference Server as the exclusive engine** for executing model predictions. Direct local model execution (e.g., executing forward passes directly via local Python model adapters) is disabled in the production pipeline.

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Consumer Pipeline
    participant EM as ExecutionManager
    participant IP as InputProcessor
    participant TM as TensorMapper
    participant TC as TritonClient
    participant TRITON as NVIDIA Triton Server (Python Backend)
    participant OP as OutputProcessor

    Caller->>EM: execute(audio_or_payload)
    EM->>IP: process(raw_input)
    IP-->>EM: preprocessed_input (e.g. 16kHz audio / text)
    EM->>TM: build_request(preprocessed_input)
    TM-->>EM: KServe v2 Tensors (AUDIO_SIGNAL [FP32], SAMPLE_RATE [INT32])
    EM->>TC: infer(model_name, inputs, outputs)
    TC->>TRITON: POST /v2/models/{model_name}/infer
    TRITON->>TRITON: Native NeMo Conformer CTC Execution
    TRITON-->>TC: HTTP 200 (TRANSCRIPT [BYTES])
    TC-->>EM: raw_response
    EM->>TM: parse_response(raw_response)
    TM-->>EM: Standard domain dict ({"task": "transcription", "text": ...})
    EM->>OP: process(raw_output)
    OP-->>EM: final_output
    EM-->>Caller: ExecutionResult(success=True, output=final_output)
```

### Key Capabilities & Option B Serving:
1. **Option B Direct NeMo Serving:** Serves native `.nemo` checkpoints directly using Triton's Python backend (`nemo_conformer_python`), completely bypassing ONNX conversion and eliminating ONNX Runtime dependencies.
2. **Zero-External-Dependency HTTP Client:** `TritonClient` implements the KServe v2 REST protocol over standard Python `urllib.request`, ensuring complete operability in air-gapped environments without requiring `tritonclient` wheels.
3. **Model Decoupled Tensor Mapping:** `TensorMapper` translates arbitrary framework inputs (audio bytes, waveforms, dictionaries, text) into typed KServe tensors and normalizes Triton responses back into domain-specific outputs.
4. **Direct Execution Protection:** `ExecutionManager._run_inference()` strictly blocks direct invocation of `model_instance.predict()`: any attempt to execute local adapter prediction without Triton raises `InferenceError`.
5. **Fail-Fast Readiness Checks:** `ExecutionManager.initialize()` validates Triton server health (`/v2/health/ready`) and model readiness (`/v2/models/{name}/ready`) during startup before accepting pipeline traffic.


