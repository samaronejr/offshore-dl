#!/bin/bash
#SBATCH --job-name=offshore-3w
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpc_%j.out
#SBATCH --error=logs/hpc_%j.err

# ==============================================================================
# hpc_job_3w.slurm — 3W full dataset production training on HPC cluster
#
# Submit from the project root:
#   sbatch scripts/hpc_job_3w.slurm
#
# IMPORTANT NOTES:
#
# 1. Partition name: The partition above is "gpu". Your NACAD cluster may use a
#    different name (e.g., "gpu_a100", "gpu-shared"). Check with: sinfo -s
#
# 2. TiRex GPU compatibility: TiRex requires CUDA compute capability >= 8.0.
#    - V100 GPUs have compute capability 7.0 — TiRex WILL FAIL on V100 nodes.
#    - A100 GPUs have compute capability 8.0 — TiRex works on A100.
#    - If only V100 nodes are available, exclude TiRex from the model list:
#        Modify the command below to add:
#        --models lstm deeponet patchtst chronos timesfm
#
# 3. HuggingFace model cache: Compute nodes may lack internet access. Pre-download
#    models on the login node BEFORE submitting this job:
#
#      export SIF_IMAGE=/path/to/offshore-dl_train.sif
#      singularity exec --nv $SIF_IMAGE python -c \
#        "from huggingface_hub import snapshot_download; \
#         snapshot_download('google/timesfm-1.0-200m-pytorch')"
#
#    Ensure HF_CACHE_DIR points to the same cache directory used at job runtime.
#
# 4. SIF image: Set SIF_IMAGE env var or place the .sif file at the project root:
#      singularity pull --name offshore-dl_train.sif docker://ghcr.io/USER/offshore-dl:train
#
# 5. 3W dataset size: The full 3W dataset contains 208k+ sliding windows. Training
#    all models may take 12-24 hours. If you hit OOM errors, consider:
#    - Reducing batch size via config overrides
#    - Requesting a node with more GPU memory (e.g., A100 80GB)
#    - Running a subset of models in separate jobs
#
# ==============================================================================

set -euo pipefail

# --- Setup -------------------------------------------------------------------
mkdir -p logs

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

# --- Job summary -------------------------------------------------------------
echo "================================================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Job Name:    ${SLURM_JOB_NAME}"
echo "Node:        $(hostname)"
echo "Start Time:  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Project Root: ${PROJECT_ROOT}"
echo "-----------------------------------------------------------------"
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "================================================================="

# --- Run experiment ----------------------------------------------------------
bash "${PROJECT_ROOT}/scripts/singularity_run.sh" \
    python scripts/run_production_3w.py --device cuda

# --- Done --------------------------------------------------------------------
echo "================================================================="
echo "End Time:    $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Exit Code:   $?"
echo "================================================================="
