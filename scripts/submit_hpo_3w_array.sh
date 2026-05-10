#!/usr/bin/env bash
# Submit Stage 1 3W classification HPO as SLURM task arrays.
#
# Examples:
#   DRY_RUN=1 bash scripts/submit_hpo_3w_array.sh
#   MAX_PARALLEL=2 N_TRIALS=30 PARTITION=gpu bash scripts/submit_hpo_3w_array.sh
#   RUN_RF_ON_CPU=1 CPU_PARTITION=cpu bash scripts/submit_hpo_3w_array.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST="${MANIFEST:-${PROJECT}/scripts/hpo_3w_models.txt}"
CAMPAIGN_ID="${CAMPAIGN_ID:-3w-hpo-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/results/hpo}"
STORAGE_DIR="${STORAGE_DIR:-${OUTPUT_DIR}/3w/${CAMPAIGN_ID}/optuna}"
LOG_DIR="${LOG_DIR:-${PROJECT}/logs}"
PARTITION="${PARTITION:-gpu}"
CPU_PARTITION="${CPU_PARTITION:-cpu}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
N_TRIALS="${N_TRIALS:-30}"
DEVICE="${DEVICE:-cuda}"
ARRAY_SCRIPT="${ARRAY_SCRIPT:-${PROJECT}/scripts/slurm_hpo_3w_array.sh}"
MANIFEST_DIR="${LOG_DIR}/hpo_manifests"
DEFAULT_MANIFEST="${PROJECT}/scripts/hpo_3w_models.txt"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "${OUTPUT_DIR}/3w/${CAMPAIGN_ID}" "${STORAGE_DIR}"

if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    RUN_MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}"
elif [[ "${USE_CONTAINER:-1}" == "1" ]]; then
    RUN_MLFLOW_TRACKING_URI="file:/app/mlruns"
else
    RUN_MLFLOW_TRACKING_URI="file:${PROJECT}/mlruns"
fi

mapfile -t ALL_MODELS < <(grep -Ev '^[[:space:]]*(#|$)' "${MANIFEST}" | sed 's/[[:space:]]//g')
if (( ${#ALL_MODELS[@]} == 0 )); then
    echo "ERROR: no active models found in ${MANIFEST}" >&2
    exit 2
fi

write_manifest() {
    local path="$1"; shift
    : >"${path}"
    for model in "$@"; do
        echo "${model}" >>"${path}"
    done
}

submit_one() {
    local manifest_path="$1"
    local partition="$2"
    local device="$3"
    local label="$4"
    local n_models
    n_models=$(awk 'NF && $1 !~ /^#/' "${manifest_path}" | wc -l | tr -d ' ')
    if (( n_models == 0 )); then
        echo "Skipping ${label}: no models"
        return 0
    fi

    local array_spec="0-$((n_models - 1))%${MAX_PARALLEL}"
    local export_vars="ALL,PROJECT=${PROJECT},MANIFEST=${manifest_path},CAMPAIGN_ID=${CAMPAIGN_ID},OUTPUT_DIR=${OUTPUT_DIR},STORAGE_DIR=${STORAGE_DIR},LOG_DIR=${LOG_DIR},N_TRIALS=${N_TRIALS},DEVICE=${device},SIF=${SIF:-${SIF_IMAGE:-}},USE_CONTAINER=${USE_CONTAINER:-1},CONTAINER_OUTPUT_DIR=${CONTAINER_OUTPUT_DIR:-/app/results/hpo},CONTAINER_STORAGE_DIR=${CONTAINER_STORAGE_DIR:-/app/results/hpo/3w/${CAMPAIGN_ID}/optuna},TRIAL_FOLDS=${TRIAL_FOLDS:-},FINAL_EVAL=${FINAL_EVAL:-1},RESUME=${RESUME:-0},FORCE=${FORCE:-0},RUN_RF_ON_CPU=${RUN_RF_ON_CPU:-0},MLFLOW_TRACKING_URI=${RUN_MLFLOW_TRACKING_URI}"

    local args=(
        --array="${array_spec}"
        --job-name="hpo-3w-${label}"
        --partition="${partition}"
        --export="${export_vars}"
    )
    if [[ "${device}" != "cpu" && -n "${GPU_GRES:-}" ]]; then
        args+=(--gres="${GPU_GRES}")
    fi
    if [[ -n "${ACCOUNT:-}" ]]; then
        args+=(--account="${ACCOUNT}")
    fi
    if [[ -n "${EXCLUDE:-}" ]]; then
        args+=(--exclude="${EXCLUDE}")
    fi
    if [[ -n "${SBATCH_EXTRA:-}" ]]; then
        # shellcheck disable=SC2206
        local extra=( ${SBATCH_EXTRA} )
        args+=("${extra[@]}")
    fi

    echo
    echo "Submit ${label}:"
    echo "  manifest: ${manifest_path}"
    echo "  models:   ${n_models}"
    echo "  array:    ${array_spec}"
    echo "  partition:${partition}"
    echo "  device:   ${device}"
    printf '  command: sbatch'
    printf ' %q' "${args[@]}" "${ARRAY_SCRIPT}"
    printf '\n'

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        return 0
    fi
    if ! command -v sbatch >/dev/null 2>&1; then
        echo "ERROR: sbatch not found; rerun with DRY_RUN=1 for preflight" >&2
        exit 127
    fi
    sbatch "${args[@]}" "${ARRAY_SCRIPT}"
}

GPU_MODELS=()
CPU_MODELS=()
if [[ "${RUN_RF_ON_CPU:-0}" == "1" ]]; then
    for model in "${ALL_MODELS[@]}"; do
        if [[ "${model}" == "random_forest" ]]; then
            CPU_MODELS+=("${model}")
        else
            GPU_MODELS+=("${model}")
        fi
    done
else
    GPU_MODELS=("${ALL_MODELS[@]}")
fi

MANIFEST_TAG=""
if [[ "$(realpath -m "${MANIFEST}")" != "$(realpath -m "${DEFAULT_MANIFEST}")" ]]; then
    MANIFEST_TAG="_$(basename "${MANIFEST}" .txt)"
fi
GPU_MANIFEST="${MANIFEST_DIR}/hpo_3w_${CAMPAIGN_ID}${MANIFEST_TAG}_gpu.txt"
CPU_MANIFEST="${MANIFEST_DIR}/hpo_3w_${CAMPAIGN_ID}${MANIFEST_TAG}_cpu.txt"
write_manifest "${GPU_MANIFEST}" "${GPU_MODELS[@]}"
write_manifest "${CPU_MANIFEST}" "${CPU_MODELS[@]}"

cat <<EOF
=================================================================
3W HPO Stage 1 submission
  campaign:     ${CAMPAIGN_ID}
  project:      ${PROJECT}
  output:       ${OUTPUT_DIR}/3w/${CAMPAIGN_ID}
  storage:      ${STORAGE_DIR}
  n_trials:     ${N_TRIALS}
  max_parallel: ${MAX_PARALLEL}
  dry_run:      ${DRY_RUN:-0}
=================================================================
EOF

submit_one "${GPU_MANIFEST}" "${PARTITION}" "${DEVICE}" "gpu"
if [[ "${RUN_RF_ON_CPU:-0}" == "1" ]]; then
    submit_one "${CPU_MANIFEST}" "${CPU_PARTITION}" "cpu" "cpu"
fi

cat <<EOF

Post-array validation command:
  python scripts/validate_hpo_3w_results.py --campaign-id ${CAMPAIGN_ID} --output-dir ${OUTPUT_DIR} --write-summary

Monitor examples:
  squeue -u "${USER:-unknown}"
  tail -f ${LOG_DIR}/hpo_3w_${CAMPAIGN_ID}_<job>_<task>.out
EOF
