# Triton NeMo Conformer-CTC Serving Container

This directory provides the reproducible container build specification and Triton model repository definition for serving NVIDIA NeMo Conformer-CTC ASR models with NVIDIA Triton Inference Server's Python backend.

## Architecture

- **Base Image:** `nvcr.io/nvidia/tritonserver:23.10-py3`
- **OS / Python:** Ubuntu 22.04 LTS / Python 3.10.12
- **Framework Versions:**
  - `torch == 2.1.2+cpu`
  - `torchaudio == 2.1.2+cpu`
  - `nemo_toolkit[asr] == 1.23.0`
  - `numpy == 1.26.4` (pinned to `< 2.0` for PyTorch 2.1.2 C-extension compatibility)
  - `soundfile == 0.14.0` (with bundled libsndfile)
  - `sentencepiece == 0.2.2`

## Model Repository Structure

```
docker/triton_nemo/models/
└── nemo_conformer_python/
    ├── config.pbtxt
    └── 1/
        └── model.py
```

- **Model Name:** `nemo_conformer_python`
- **Backend:** `python`
- **Input Tensors:**
  - `AUDIO_SIGNAL`: FP32, shape `[-1]` (raw 1D audio waveform)
  - `SAMPLE_RATE`: INT32, shape `[1]` (input sample rate, auto-resampled to 16 kHz)
- **Output Tensor:**
  - `TRANSCRIPT`: TYPE_STRING, shape `[1]` (transcribed text)

## Building the Image

Run from the `ml_framework` root directory:

```bash
docker build -t tritonserver-nemo:23.10-cpu -f docker/triton_nemo/Dockerfile docker/triton_nemo/
```

## Running the Serving Container

Launch the container with the reproducible Triton model repository and model checkpoint directory mounted:

```bash
docker run -d \
  --name triton_conformer_reproducible \
  -p 8010:8000 -p 8011:8001 -p 8012:8002 \
  -v $(pwd)/docker/triton_nemo/models:/models \
  -v /path/to/models:/model \
  tritonserver-nemo:23.10-cpu \
  tritonserver --model-repository=/models
```

Notice: Unlike the transient POC setup, this image **does not** require mounting `/tmp/poc_nemo_pkgs/site-packages` or configuring `PYTHONPATH=/poc_pkgs`. All Python dependencies (`torch`, `torchaudio`, `nemo_toolkit[asr]`, `soundfile`) are self-contained natively inside `/usr/local/lib/python3.10/dist-packages`.

## Runtime Boundary & Dependency Architecture

A strict architectural separation is maintained between the Triton serving container and the client framework:

1. **Triton Serving Container:**
   - Dedicated strictly to model forward pass inference execution.
   - Communicates with callers over KServe v2 protocol (HTTP / gRPC) via `AUDIO_SIGNAL` and `SAMPLE_RATE` tensors.
   - Loads model checkpoints from mounted filesystems (`/model` or `/models`).
   - Does **not** ingest from Kafka or MinIO directly.
   - Does **not** contain `confluent-kafka` or `boto3`.

2. **Framework Client Runtime (`ml_framework`):**
   - Reusable Python library executing in host applications, workers, or stream processing services.
   - Manages data ingestion from Kafka topics (`confluent-kafka`) and S3/MinIO/SeaweedFS storage (`boto3`).
   - Performs audio decoding, resampling, and tensor preparation.
   - Dispatches inference requests to Triton via standard HTTP (`TritonClient` / `urllib.request`).
   - Connector dependencies are defined in `pyproject.toml` (`[project.optional-dependencies]`) and installed in the framework runtime (`pip install 'ml-framework[connectors]'`).
