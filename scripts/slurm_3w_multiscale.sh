#!/bin/bash
#SBATCH --job-name=odl-multiscale
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/3w_multiscale_%j.out
#SBATCH --error=logs/3w_multiscale_%j.err

echo "=== Multi-scale Features (RF + DeepONet) 3W === Node: $(hostname) Date: $(date)"
nvidia-smi -L
PROJECT=~/offshore-dl
SIF=$PROJECT/offshore-dl-train.sif

singularity exec --nv --no-home \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --bind $PROJECT/reports:/app/reports \
    --bind $PROJECT/scripts:/app/scripts \
    --bind $PROJECT/src:/app/src \
    --bind $PROJECT/configs:/app/configs \
    --bind ${HOME}/.cache/huggingface:/app/.cache/huggingface \
    --pwd /app $SIF \
    python scripts/run_production_3w_features.py --device cuda --models multiscale_rf multiscale_deeponet

echo "=== Multi-scale DONE: $(date) ==="
