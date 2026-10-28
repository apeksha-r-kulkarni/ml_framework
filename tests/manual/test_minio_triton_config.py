
import os
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_framework import ml, FrameworkConfig, StorageConfig, TritonConfig

# Selected model configuration (NeMo Conformer CTC Large)
MODEL_BUCKET = os.environ.get("MINIO_MODEL_BUCKET", "speech-data")
MODEL_KEY = os.environ.get("MINIO_MODEL_KEY", "model/Transcription/nemo_conformer_ctc/stt_en_conformer_ctc_large.nemo")
TRITON_MODEL_NAME = os.environ.get("TRITON_MODEL_NAME", "nemo_conformer_python")
TRITON_URL = os.environ.get("TRITON_SERVER_URL", "http://localhost:8010")

# Audio input source: prefer MinIO storage URI with local test WAV fallback
AUDIO_FILE = os.environ.get("INPUT_AUDIO_SOURCE", "/tmp/poc_nemo_audio/test3.wav")
MINIO_AUDIO_URI = "s3://media-audio/09_2026/08/audio/wav/test3.wav"

config = FrameworkConfig(
    model_bucket=MODEL_BUCKET,
    model_key=MODEL_KEY,
    model_name=TRITON_MODEL_NAME,
    storage_config=StorageConfig.from_env(),
    triton_config=TritonConfig(
        endpoint_url=TRITON_URL,
        model_name=TRITON_MODEL_NAME,
        backend="python",
        staging_dir="/tmp/poc_nemo_model",
    ),
)

manager = None

try:
    # Determine effective audio input
    effective_audio = AUDIO_FILE
    if not (effective_audio.startswith(("s3://", "minio://")) or os.path.isfile(effective_audio)):
        if os.path.isfile("/tmp/poc_nemo_audio/test3.wav"):
            effective_audio = "/tmp/poc_nemo_audio/test3.wav"
        else:
            effective_audio = MINIO_AUDIO_URI

    print("Selected MinIO bucket:", MODEL_BUCKET)
    print("Selected MinIO object:", MODEL_KEY)
    print("Selected Triton model:", TRITON_MODEL_NAME)
    print("Input audio source  :", effective_audio)

    start = time.perf_counter()
    manager = ml.load(config=config)
    print(f"Framework initialization: {time.perf_counter() - start:.2f}s")
    print("Framework ready:", manager.is_ready)

    start = time.perf_counter()
    result = manager.execute(effective_audio)
    print(f"Inference duration: {time.perf_counter() - start:.2f}s")

    print("Execution success:", result.success)
    print("Execution state:", result.state)

    if not result.success:
        print("Execution error:", result.error)
        sys.exit(1)

    print("Model output:")
    print(result.output)

except Exception as exc:
    print(f"TEST FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)

finally:
    if manager is not None:
        # Use the framework's supported lifecycle API for cleanup.
        try:
            manager.stop()
        except (AttributeError, TypeError):
            pass
