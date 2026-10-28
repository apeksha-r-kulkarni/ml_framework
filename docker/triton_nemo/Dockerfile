# Self-contained NVIDIA Triton Inference Server for NeMo Conformer CTC ASR Serving
# Base: NVIDIA Triton 23.10 (Ubuntu 22.04 LTS, Python 3.10.12)
FROM nvcr.io/nvidia/tritonserver:23.10-py3

LABEL maintainer="Shyena ML Team <apeksha.kulkarni@shyena.in>"
LABEL description="Triton 23.10 image with pinned PyTorch CPU and NeMo ASR dependencies for Conformer-CTC serving"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install native system audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    sox \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*

# Step 1: Install build tools and numpy pinned < 2.0 (PyTorch 2.1.2 requires numpy 1.x)
RUN pip install --no-cache-dir Cython==3.3.0 numpy==1.26.4

# Step 2: Install PyTorch CPU wheels
RUN pip install --no-cache-dir torch==2.1.2+cpu torchaudio==2.1.2+cpu --extra-index-url https://download.pytorch.org/whl/cpu

# Step 3: Install pinned NeMo ASR and supporting dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Step 4: Verification smoke check at build time
RUN python3 -c "import torch, torchaudio, nemo.collections.asr as nemo_asr, soundfile; print('NeMo Triton Environment Verified: torch=' + torch.__version__ + ', nemo=' + str(nemo_asr.__file__))"

WORKDIR /opt/tritonserver
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
CMD ["tritonserver", "--model-repository=/models"]
