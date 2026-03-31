#!/bin/bash
#SBATCH --job-name=odl-tirex
#SBATCH --partition=gpu
#SBATCH --exclude=caloba94
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tirex_%j.out
#SBATCH --error=logs/tirex_%j.err

echo "=== TiRex 3W: Extract Embeddings + RF Classification ==="
echo "Node: $(hostname), Date: $(date)"
nvidia-smi -L

PROJECT=~/offshore-dl
SIF=$PROJECT/offshore-dl-train.sif

# Step 1: Extract embeddings to memmap (GPU, ~30 min)
echo "--- Step 1: Extracting embeddings ---"
singularity exec --nv \
    --no-home \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --bind ${HOME}/.cache/huggingface:/app/.cache/huggingface \
    --pwd /app \
    $SIF \
    python scripts/extract_tirex_embeddings.py --device cuda

echo "--- Step 1 done: $(date) ---"

# Step 2: Run RF nested evaluation (CPU, ~10 min)
echo "--- Step 2: RF nested evaluation ---"
singularity exec --nv \
    --no-home \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --bind ${HOME}/.cache/huggingface:/app/.cache/huggingface \
    --pwd /app \
    $SIF \
    python scripts/run_tirex_rf_nested.py

echo "=== TiRex DONE: $(date) ==="
