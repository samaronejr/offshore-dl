#!/bin/bash
#SBATCH --job-name=odl-gan-chronos-ft
#SBATCH --partition=gpu
#SBATCH --nodelist=caloba94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gan_chronos_finetune_%j.out
#SBATCH --error=logs/gan_chronos_finetune_%j.err

set -euo pipefail

echo "=== Ganymede Chronos-2 fine-tuning ==="
echo "Node: $(hostname), Date: $(date)"
nvidia-smi -L

PROJECT="${PROJECT:-$HOME/offshore-dl}"
SIF="${SIF_IMAGE:-$PROJECT/offshore-dl-train.sif}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

mkdir -p "$PROJECT/results" "$PROJECT/logs" "$HF_CACHE_DIR"

singularity exec --nv \
    --no-home \
    --writable-tmpfs \
    --bind "$PROJECT/scripts:/app/scripts" \
    --bind "$PROJECT/src:/app/src" \
    --bind "$PROJECT/configs:/app/configs" \
    --bind "$PROJECT/data:/app/data" \
    --bind "$PROJECT/results:/app/results" \
    --bind "$HF_CACHE_DIR:/app/.cache/huggingface" \
    --pwd /app \
    "$SIF" \
    bash -c "pip install --quiet --target /tmp/pip_pkgs peft && PYTHONPATH=/tmp/pip_pkgs:\$PYTHONPATH python scripts/run_ganymede_chronos_finetune.py --device cuda $*"

echo "=== Ganymede Chronos-2 fine-tuning done: $(date) ==="
