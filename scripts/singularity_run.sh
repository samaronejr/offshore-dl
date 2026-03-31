#!/usr/bin/env bash
# singularity_run.sh — Convenience wrapper for running offshore-dl in Singularity/Apptainer
#
# Mirrors scripts/docker_run.sh with Singularity-equivalent flags.
#
# Usage:
#   scripts/singularity_run.sh pytest tests/ --tb=short
#   scripts/singularity_run.sh python scripts/run_production_ganymede.py --device cuda
#   scripts/singularity_run.sh bash          # interactive shell
#
# Bind mounts (mirrors docker_run.sh volumes):
#   data/      → /app/data       (read-only input data)
#   results/   → /app/results    (experiment outputs)
#   mlruns/    → /app/mlruns     (MLflow tracking)
#   reports/   → /app/reports    (generated reports)
#   HF cache   → /app/.cache/huggingface (model downloads)
#
# Environment variables:
#   SIF_IMAGE     — Path to .sif file (default: $PROJECT_ROOT/offshore-dl_train.sif)
#   HF_TOKEN      — HuggingFace token (passed via runtime-appropriate env prefix)
#   HF_CACHE_DIR  — Host HuggingFace cache directory (default: ~/.cache/huggingface)

set -euo pipefail

# --- Detect container runtime ------------------------------------------------
if command -v apptainer &>/dev/null; then
    RUNTIME="apptainer"
    ENV_PREFIX="APPTAINERENV"
elif command -v singularity &>/dev/null; then
    RUNTIME="singularity"
    ENV_PREFIX="SINGULARITYENV"
else
    echo "✗ Error: Neither 'apptainer' nor 'singularity' found in PATH." >&2
    echo "  Install Apptainer: https://apptainer.org/docs/admin/main/installation.html" >&2
    exit 1
fi
echo "▸ Container runtime: ${RUNTIME}"

# --- Project root (same derivation as docker_run.sh) -------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- SIF image path ----------------------------------------------------------
SIF_PATH="${SIF_IMAGE:-${PROJECT_ROOT}/offshore-dl_train.sif}"
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "✗ Error: SIF image not found at ${SIF_PATH}" >&2
    echo "  Pull it:  ${RUNTIME} pull docker://ghcr.io/YOUR_USER/offshore-dl:train" >&2
    echo "  Or set:   SIF_IMAGE=/path/to/image.sif" >&2
    exit 1
fi

# --- Bind mounts (mirrors docker_run.sh volumes) ----------------------------
BIND_FLAGS=(
    --bind "${PROJECT_ROOT}/data:/app/data:ro"
    --bind "${PROJECT_ROOT}/results:/app/results"
    --bind "${PROJECT_ROOT}/mlruns:/app/mlruns"
    --bind "${PROJECT_ROOT}/reports:/app/reports"
    --bind "${HF_CACHE_DIR:-${HOME}/.cache/huggingface}:/app/.cache/huggingface"
)

# --- Ensure writable output directories exist --------------------------------
mkdir -p "${PROJECT_ROOT}/results" "${PROJECT_ROOT}/mlruns" "${PROJECT_ROOT}/reports"

# --- HuggingFace token passthrough -------------------------------------------
# Singularity/Apptainer pass env vars into the container via a prefix convention:
#   SINGULARITYENV_VAR=val → container sees VAR=val
#   APPTAINERENV_VAR=val   → container sees VAR=val
if [[ -n "${HF_TOKEN:-}" ]]; then
    export "${ENV_PREFIX}_HF_TOKEN=${HF_TOKEN}"
fi

# --- Execute ------------------------------------------------------------------
echo "▸ Running: ${RUNTIME} exec --nv --no-home ${SIF_PATH} $*"
echo "▸ Project root: ${PROJECT_ROOT}"

exec "${RUNTIME}" exec \
    --nv \
    --no-home \
    "${BIND_FLAGS[@]}" \
    "${SIF_PATH}" \
    "$@"
