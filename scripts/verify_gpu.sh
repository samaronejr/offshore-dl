#!/usr/bin/env bash
# GPU Verification Script — run on a machine with NVIDIA GPU + Docker + nvidia-container-toolkit
#
# Prerequisites:
#   docker build -t offshore-dl:train -f docker/Dockerfile .
#
# Usage:
#   scripts/verify_gpu.sh
#
# Expected output:
#   1. torch.cuda.is_available() = True
#   2. GPU name and memory
#   3. LSTM trains on CDF with gpu_memory_peak_mb > 0
#   4. Result JSON written to results/lstm/cdf.json

set -euo pipefail

IMAGE="${OFFSHORE_DL_IMAGE:-offshore-dl:train}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "═══════════════════════════════════════════════════"
echo "  GPU Verification — offshore-dl Docker Pipeline"
echo "═══════════════════════════════════════════════════"

# Step 1: Check CUDA
echo ""
echo ">>> Step 1: Verify CUDA inside container"
docker run --rm --gpus all "${IMAGE}" python -c "
import torch
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'  Compute capability: {torch.cuda.get_device_capability()}')
else:
    print('  ERROR: CUDA not available inside container!')
    exit(1)
"

# Step 2: Run LSTM on CDF with GPU
echo ""
echo ">>> Step 2: Train LSTM on CDF (GPU, 3 epochs)"
docker run --rm --gpus all \
    -v "${PROJECT_ROOT}/data:/app/data:ro" \
    -v "${PROJECT_ROOT}/results:/app/results" \
    "${IMAGE}" \
    python -m offshore_dl.run_experiment \
        --model lstm --dataset cdf --device cuda --max-epochs 3 --no-mlflow

# Step 3: Check result
echo ""
echo ">>> Step 3: Verify GPU metrics in result JSON"
python3 -c "
import json, sys
r = json.load(open('results/lstm/cdf.json'))
gpu_mb = r.get('cost', {}).get('gpu_memory_peak_mb_mean', 0)
wall_s = r.get('cost', {}).get('wall_time_seconds_mean', 0)
print(f'  gpu_memory_peak_mb_mean: {gpu_mb:.1f}')
print(f'  wall_time_seconds_mean: {wall_s:.2f}')
if gpu_mb > 0:
    print('  ✓ GPU training verified!')
else:
    print('  ✗ GPU memory is 0 — training may not have used GPU')
    sys.exit(1)
"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  GPU verification complete"
echo "═══════════════════════════════════════════════════"
