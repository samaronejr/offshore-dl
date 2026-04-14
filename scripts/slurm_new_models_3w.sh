#!/bin/bash
#SBATCH --job-name=odl-new-models-3w
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/new_models_3w_%j.out
#SBATCH --error=logs/new_models_3w_%j.err

# Evaluate new architectures on 3W classification:
# - ConvTran (raw 720x27 + features 14x27)
# - InceptionTime (raw 720x27 + features 14x27)
# - Multi-scale features (180+360+720 -> 42x27) with RF and DeepONet
# - Focal Loss variants (LSTM + DeepONet with focal loss)
# - Hydra+MultiROCKET (requires aeon: pip install aeon)

echo "=== New Models 3W Evaluation ==="
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
    python scripts/run_production_3w_features.py \
        --device cuda \
        --models convtran convtran_raw inception_time inception_time_raw \
                 lstm_focal deeponet_focal \
                 multiscale_rf multiscale_deeponet

echo "=== New Models 3W DONE: $(date) ==="
