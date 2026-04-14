#!/bin/bash
#SBATCH --job-name=odl-rerun-inner-mongolia
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/rerun_inner_mongolia_%j.out
#SBATCH --error=logs/rerun_inner_mongolia_%j.err

echo "=== Inner Mongolia Rerun (post-preprocessing-fix) ==="
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
    python scripts/run_production_inner_mongolia.py --device cuda

echo "=== Inner Mongolia Rerun DONE: $(date) ==="
