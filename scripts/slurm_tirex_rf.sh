#!/bin/bash
#SBATCH --job-name=odl-tirex-rf
#SBATCH --partition=gpu
#SBATCH --exclude=caloba94
#SBATCH --nodelist=caloba91
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=28G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tirex_rf_%j.out
#SBATCH --error=logs/tirex_rf_%j.err

echo "=== TiRex 3W: RF Nested Evaluation (embeddings pre-extracted) ==="
echo "Node: $(hostname), Date: $(date)"

PROJECT=~/offshore-dl
SIF=$PROJECT/offshore-dl-train.sif

singularity exec --nv \
    --no-home \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --bind ${HOME}/.cache/huggingface:/app/.cache/huggingface \
    --pwd /app \
    $SIF \
    python scripts/run_tirex_rf_nested.py

echo "=== TiRex RF DONE: $(date) ==="
