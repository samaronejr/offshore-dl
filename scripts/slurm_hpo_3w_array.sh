#!/usr/bin/env bash
#SBATCH --job-name=hpo-3w
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/hpo_3w_%A_%a.sbatch.out
#SBATCH --error=logs/hpo_3w_%A_%a.sbatch.err

# Stage 1 3W classification HPO array task.
# Submit via scripts/submit_hpo_3w_array.sh so CAMPAIGN_ID and array bounds are set.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST="${MANIFEST:-${PROJECT}/scripts/hpo_3w_models.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/results/hpo}"
CAMPAIGN_ID="${CAMPAIGN_ID:-3w-hpo-$(date -u +%Y%m%dT%H%M%SZ)}"
STORAGE_DIR="${STORAGE_DIR:-${OUTPUT_DIR}/3w/${CAMPAIGN_ID}/optuna}"
LOG_DIR="${LOG_DIR:-${PROJECT}/logs}"
N_TRIALS="${N_TRIALS:-30}"
DEVICE="${DEVICE:-cuda}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}/3w/${CAMPAIGN_ID}" "${STORAGE_DIR}"
LOG_OUT="${LOG_DIR}/hpo_3w_${CAMPAIGN_ID}_${JOB_ID}_${TASK_ID}.out"
LOG_ERR="${LOG_DIR}/hpo_3w_${CAMPAIGN_ID}_${JOB_ID}_${TASK_ID}.err"
exec >"${LOG_OUT}" 2>"${LOG_ERR}"

mapfile -t MODELS < <(grep -Ev '^[[:space:]]*(#|$)' "${MANIFEST}" | sed 's/[[:space:]]//g')
N_MODELS="${#MODELS[@]}"
if (( TASK_ID < 0 || TASK_ID >= N_MODELS )); then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} outside manifest range 0-$((N_MODELS - 1))" >&2
    exit 2
fi

MODEL="${MODELS[$TASK_ID]}"
if [[ "${MODEL}" == "random_forest" && "${RUN_RF_ON_CPU:-0}" == "1" ]]; then
    DEVICE="${RF_DEVICE:-cpu}"
fi

RUN_OUTPUT_DIR="${OUTPUT_DIR}"
RUN_STORAGE_DIR="${STORAGE_DIR}"
RUN_MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:${PROJECT}/mlruns}"
if [[ "${USE_CONTAINER:-1}" == "1" ]]; then
    RUN_OUTPUT_DIR="${CONTAINER_OUTPUT_DIR:-/app/results/hpo}"
    RUN_STORAGE_DIR="${CONTAINER_STORAGE_DIR:-/app/results/hpo/3w/${CAMPAIGN_ID}/optuna}"
    RUN_MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file:/app/mlruns}"
fi

CMD=(
    python scripts/run_optuna_hpo.py
    --dataset 3w
    --models "${MODEL}"
    --n-trials "${N_TRIALS}"
    --device "${DEVICE}"
    --campaign-id "${CAMPAIGN_ID}"
    --output-dir "${RUN_OUTPUT_DIR}"
    --storage-dir "${RUN_STORAGE_DIR}"
    --skip-existing
    --no-summary
)

if [[ "${FORCE:-0}" == "1" ]]; then
    CMD+=(--force)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    CMD+=(--resume)
fi
if [[ -n "${TRIAL_FOLDS:-}" ]]; then
    CMD+=(--trial-folds "${TRIAL_FOLDS}")
fi
if [[ "${FINAL_EVAL:-1}" == "0" ]]; then
    CMD+=(--no-final-eval)
else
    CMD+=(--final-eval)
fi

cat <<EOF
=================================================================
3W HPO array task
  campaign: ${CAMPAIGN_ID}
  job/task: ${JOB_ID}/${TASK_ID}
  model:    ${MODEL}
  node:     $(hostname)
  start:    $(date -Is)
  project:  ${PROJECT}
  output:   ${OUTPUT_DIR} (runtime: ${RUN_OUTPUT_DIR})
  storage:  ${STORAGE_DIR} (runtime: ${RUN_STORAGE_DIR})
  log out:  ${LOG_OUT}
  log err:  ${LOG_ERR}
=================================================================
EOF

nvidia-smi -L || true
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${HPO_DRY_RUN:-0}" == "1" ]]; then
    echo "HPO_DRY_RUN=1; not executing."
    exit 0
fi

cd "${PROJECT}"
export MLFLOW_TRACKING_URI="${RUN_MLFLOW_TRACKING_URI}"

if [[ "${USE_CONTAINER:-1}" == "1" ]]; then
    DEFAULT_SIF_IMAGE="${PROJECT}/offshore-dl_train.sif"
    if [[ ! -f "${DEFAULT_SIF_IMAGE}" && -f "${PROJECT}/offshore-dl-train.sif" ]]; then
        DEFAULT_SIF_IMAGE="${PROJECT}/offshore-dl-train.sif"
    fi
    export SIF_IMAGE="${SIF:-${SIF_IMAGE:-${DEFAULT_SIF_IMAGE}}}"
    bash "${PROJECT}/scripts/singularity_run.sh" "${CMD[@]}"
else
    "${CMD[@]}"
fi

cat <<EOF
=================================================================
3W HPO task done
  model: ${MODEL}
  end:   $(date -Is)
=================================================================
EOF
