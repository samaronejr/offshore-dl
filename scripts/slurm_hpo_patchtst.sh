#!/bin/bash
#SBATCH --job-name=hpo-patchtst
#SBATCH --partition=gpu
#SBATCH --exclude=caloba94,caloba91
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-12:00:00
#SBATCH --output=logs/hpo_patchtst_%j.out
#SBATCH --error=logs/hpo_patchtst_%j.err

echo "=== HPO PatchTST 3W (30 trials) ==="
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
    python scripts/run_optuna_hpo.py --dataset 3w --models patchtst --n-trials 30 --device cuda

echo "=== HPO PatchTST DONE: $(date) ==="
