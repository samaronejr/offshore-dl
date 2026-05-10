#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00

# Run one forecasting MASE-repair rerun slice from a TSV manifest.
# Submit example:
#   sbatch --array=0-307%6 scripts/slurm_forecasting_mase_rerun_array.sh logs/forecasting_mase_manifest.tsv

set -euo pipefail

MANIFEST="${1:?Usage: slurm_forecasting_mase_rerun_array.sh <manifest.tsv>}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
PROJECT="${PROJECT:-${HOME}/offshore-dl}"
SIF="${SIF:-${PROJECT}/offshore-dl-train.sif}"
LOG_DIR="${LOG_DIR:-${PROJECT}/logs}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT}/results/post_fix}"
case "${RESULTS_ROOT}" in
    /*) ;;
    *) RESULTS_ROOT="${PROJECT}/${RESULTS_ROOT}" ;;
esac

mkdir -p "${LOG_DIR}" "${RESULTS_ROOT}"
exec >"${LOG_DIR}/forecasting_mase_${JOB_ID}_${TASK_ID}.out" \
     2>"${LOG_DIR}/forecasting_mase_${JOB_ID}_${TASK_ID}.err"

cd "${PROJECT}"

LINE=$(awk -v n=$((TASK_ID + 2)) 'NR == n {gsub(/\r$/, ""); print; found=1} END {if (!found) exit 42}' "${MANIFEST}") || {
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} outside manifest ${MANIFEST}" >&2
    exit 42
}
IFS=$'\t' read -r DATASET MODEL HORIZON MODES WELLS <<<"${LINE}"

case "${DATASET}" in
    ganymede)       SCRIPT="scripts/run_production_ganymede.py" ;;
    spe_berg)       SCRIPT="scripts/run_production_spe_berg.py" ;;
    volve)          SCRIPT="scripts/run_production_volve.py" ;;
    inner_mongolia) SCRIPT="scripts/run_production_inner_mongolia.py" ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'" >&2
        exit 2
        ;;
esac

CMD=(python "${SCRIPT}" --device cuda --no-mlflow --models "${MODEL}" --horizons "${HORIZON}")

IFS=',' read -ra MODE_LIST <<<"${MODES}"
if [[ ${#MODE_LIST[@]} -gt 0 && -n "${MODE_LIST[0]}" ]]; then
    CMD+=(--modes "${MODE_LIST[@]}")
fi

IFS=',' read -ra WELL_LIST <<<"${WELLS}"
if [[ ${#WELL_LIST[@]} -gt 0 && -n "${WELL_LIST[0]}" ]]; then
    CMD+=(--wells "${WELL_LIST[@]}")
fi

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:/tmp/mlruns-${USER}}"
export OFFSHORE_DL_RESULTS_DIR="${OFFSHORE_DL_RESULTS_DIR:-/app/results}"

printf '=== Forecasting MASE rerun slice ===\n'
printf 'Job: %s task %s\n' "${JOB_ID}" "${TASK_ID}"
printf 'Node: %s\n' "$(hostname)"
printf 'Date: %s\n' "$(date)"
printf 'Dataset=%s Model=%s Horizon=%s Modes=%s Wells=%s\n' \
    "${DATASET}" "${MODEL}" "${HORIZON}" "${MODES}" "${WELLS:-<none>}"
nvidia-smi -L || true
printf 'Command: %q ' "${CMD[@]}"; printf '\n'

singularity exec --nv \
    --no-home \
    --bind "${PROJECT}/data:/app/data" \
    --bind "${RESULTS_ROOT}:/app/results" \
    --bind "${PROJECT}/reports:/app/reports" \
    --bind "${PROJECT}/src:/app/src" \
    --bind "${PROJECT}/scripts:/app/scripts" \
    --bind "${PROJECT}/configs:/app/configs" \
    --bind "${HOME}/.cache/huggingface:/app/.cache/huggingface" \
    --pwd /app \
    "${SIF}" \
    "${CMD[@]}"

printf '=== DONE %s ===\n' "$(date)"
