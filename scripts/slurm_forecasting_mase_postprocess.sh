#!/bin/bash
#SBATCH --job-name=forecast-mase-post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/forecasting_mase_post_%j.out
#SBATCH --error=logs/forecasting_mase_post_%j.err

set -euo pipefail

PROJECT="${PROJECT:-${HOME}/offshore-dl}"
SIF="${SIF:-${PROJECT}/offshore-dl-train.sif}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT}/results/post_fix}"
case "${RESULTS_ROOT}" in
    /*) ;;
    *) RESULTS_ROOT="${PROJECT}/${RESULTS_ROOT}" ;;
esac
cd "${PROJECT}"

mkdir -p logs reports "${RESULTS_ROOT}"

echo "=== Forecasting MASE postprocess ==="
echo "Node: $(hostname), Date: $(date)"

singularity exec \
    --no-home \
    --bind "${PROJECT}/data:/app/data" \
    --bind "${RESULTS_ROOT}:/app/results" \
    --bind "${PROJECT}/reports:/app/reports" \
    --bind "${PROJECT}/src:/app/src" \
    --bind "${PROJECT}/scripts:/app/scripts" \
    --bind "${PROJECT}/configs:/app/configs" \
    --pwd /app \
    "${SIF}" \
    bash -lc 'OFFSHORE_DL_RESULTS_DIR=/app/results python scripts/aggregate_forecasting_results.py && python -m offshore_dl.analysis.forecasting_audit --results-dir /app/results'

echo "=== Forecasting MASE postprocess DONE: $(date) ==="
