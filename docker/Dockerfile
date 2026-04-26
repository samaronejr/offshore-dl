# ─── Stage 1: Base (Python + PyTorch + scientific stack) ───
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# System deps + Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv (avoids --break-system-packages issues on Ubuntu 22.04 pip)
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core Python deps
COPY pyproject.toml .
COPY src/ src/
RUN pip install --upgrade pip && \
    pip install -e .

# ─── Stage 2: Train (adds FM deps + configs + tests) ───
FROM base AS train

# Install foundation model + dev dependencies (pytest, etc.)
RUN pip install -e ".[fm,dev]"
RUN pip install aeon peft lightgbm

# State-space models (Mamba) — required by fkmad and mambasl. Prebuilt wheels
# from PyPI match the CUDA 12.4 runtime in the base image. --no-build-isolation
# avoids pip trying to re-resolve torch and forcing a source build (which would
# fail because this image uses cudnn-runtime and has no nvcc).
RUN pip install --no-build-isolation \
        "causal-conv1d>=1.4.0" \
        "mamba-ssm>=2.2.0"

# Copy configs, scripts, tests
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/
COPY dvc.yaml .

# HuggingFace cache directory (mount or pre-populate at runtime)
ENV HF_HOME=/app/.cache/huggingface

# Default: run data availability check
CMD ["python", "-m", "offshore_dl.data.check"]
