#!/bin/bash
#SBATCH --job-name=odl-gan-lightgbm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/gan_lightgbm_%j.out
#SBATCH --error=logs/gan_lightgbm_%j.err

echo "=== Ganymede LightGBM Baseline ==="
echo "Node: $(hostname), Date: $(date)"
nvidia-smi -L

PROJECT=~/offshore-dl
SIF=$PROJECT/offshore-dl-train.sif

singularity exec --nv \
    --no-home \
    --bind $PROJECT/scripts:/app/scripts \
    --bind $PROJECT/src:/app/src \
    --bind $PROJECT/configs:/app/configs \
    --bind $PROJECT/data:/app/data \
    --bind $PROJECT/results:/app/results \
    --pwd /app \
    $SIF \
    python scripts/run_ganymede_lightgbm.py --horizons 7 14 30 90

echo "=== Ganymede LightGBM DONE: $(date) ==="
