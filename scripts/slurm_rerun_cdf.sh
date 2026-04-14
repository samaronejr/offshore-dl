#!/bin/bash
#SBATCH --job-name=odl-rerun-cdf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rerun_cdf_%j.out
#SBATCH --error=logs/rerun_cdf_%j.err

echo "=== CDF Rerun (FM normalization fix: C3/C4) ==="
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
    python scripts/run_production_cdf.py --device cuda

echo "=== CDF Rerun DONE: $(date) ==="
