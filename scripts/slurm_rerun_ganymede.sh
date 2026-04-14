#!/bin/bash
#SBATCH --job-name=odl-rerun-ganymede
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/rerun_ganymede_%j.out
#SBATCH --error=logs/rerun_ganymede_%j.err

# Rerun Ganymede trained models after preprocessing fix (C2: removed bfill, causal shutdown).
# Reruns LSTM, DeepONet, PatchTST, TCN only (FMs are zero-shot, unaffected by preprocessing).
# Existing result files are overwritten (--skip-existing is NOT passed).

echo "=== Ganymede Rerun (post-preprocessing-fix) ==="
echo "Node: $(hostname), Date: $(date)"
nvidia-smi -L

PROJECT=~/offshore-dl
SIF=$PROJECT/offshore-dl-train.sif

singularity exec --nv \
    --no-home \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --bind $PROJECT/reports:/app/reports \
    --bind ${HOME}/.cache/huggingface:/app/.cache/huggingface \
    --pwd /app \
    $SIF \
    python scripts/run_production_ganymede.py \
        --device cuda \
        --models lstm deeponet patchtst tcn

echo "=== Ganymede Rerun DONE: $(date) ==="
