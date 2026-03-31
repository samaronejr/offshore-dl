#!/bin/bash
#SBATCH --job-name=hpo-deeponet
#SBATCH --partition=gpu
#SBATCH --exclude=caloba94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/hpo_deeponet_%j.out
#SBATCH --error=logs/hpo_deeponet_%j.err

echo "=== HPO DeepONet 3W (30 trials) ==="
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
    python scripts/run_optuna_hpo.py --dataset 3w --models deeponet --n-trials 30 --device cuda

echo "=== HPO DeepONet DONE: $(date) ==="
