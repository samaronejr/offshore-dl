#!/usr/bin/env bash
# docker_run.sh — Convenience wrapper for running offshore-dl in Docker
#
# Usage:
#   scripts/docker_run.sh pytest tests/ --tb=short
#   scripts/docker_run.sh python -m offshore_dl.run_experiment --model lstm --dataset cdf --device cuda
#   scripts/docker_run.sh bash          # interactive shell
#   scripts/docker_run.sh               # default CMD (data check)
#
# Volumes mounted:
#   data/      → /app/data       (read-only input data)
#   results/   → /app/results    (experiment outputs)
#   mlruns/    → /app/mlruns     (MLflow tracking)
#   reports/   → /app/reports    (generated reports)
#   HF cache   → /app/.cache/huggingface (model downloads)

set -euo pipefail

IMAGE="${DOCKER_IMAGE:-offshore-dl:train}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Build volume mount flags
VOLUMES=(
    -v "${PROJECT_ROOT}/data:/app/data"
    -v "${PROJECT_ROOT}/results:/app/results"
    -v "${PROJECT_ROOT}/mlruns:/app/mlruns"
    -v "${PROJECT_ROOT}/reports:/app/reports"
    -v "${HF_CACHE_DIR:-${HOME}/.cache/huggingface}:/app/.cache/huggingface"
)

# GPU passthrough (requires nvidia-container-toolkit)
GPU_FLAGS="--gpus all"

# Pass HuggingFace token if set in host environment
ENV_FLAGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    ENV_FLAGS+=(-e "HF_TOKEN=${HF_TOKEN}")
fi
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    ENV_FLAGS+=(-e "MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}")
fi

# Detect interactive mode
TTY_FLAGS=""
if [[ -t 0 ]]; then
    TTY_FLAGS="-it"
fi

echo "▸ Running: docker run ${GPU_FLAGS} ${IMAGE} $*"
echo "▸ Project root: ${PROJECT_ROOT}"

exec docker run --rm \
    ${TTY_FLAGS} \
    ${GPU_FLAGS} \
    --shm-size=16g \
    "${VOLUMES[@]}" \
    "${ENV_FLAGS[@]+"${ENV_FLAGS[@]}"}" \
    "${IMAGE}" \
    "$@"
