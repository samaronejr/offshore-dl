#!/bin/bash
# Deploy committed forecasting MASE repair code and submit controlled reruns to SLURM.

set -euo pipefail

REMOTE="${REMOTE:-LPS_loginServer}"
REMOTE_DIR="${REMOTE_DIR:-~/offshore-dl}"
THROTTLE="${THROTTLE:-6}"
WELLS_PER_CHUNK="${WELLS_PER_CHUNK:-10}"
RESULTS_ROOT="${RESULTS_ROOT:-results/post_fix}"
CAMPAIGN_ID="${CAMPAIGN_ID:-forecast-mase-$(date -u +%Y%m%dT%H%M%SZ)}"
LOCAL_MANIFEST="logs/${CAMPAIGN_ID}_manifest.tsv"
REMOTE_MANIFEST="logs/${CAMPAIGN_ID}_manifest.tsv"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: run from the offshore-dl git checkout" >&2
    exit 2
fi

python scripts/build_forecasting_mase_manifest.py \
    --output "${LOCAL_MANIFEST}" \
    --wells-per-chunk "${WELLS_PER_CHUNK}"
TASKS=$(( $(wc -l < "${LOCAL_MANIFEST}") - 1 ))
if (( TASKS < 1 )); then
    echo "ERROR: manifest has no tasks" >&2
    exit 3
fi
ARRAY_SPEC="0-$((TASKS - 1))%${THROTTLE}"

printf '═══ Deploying committed tree to %s:%s ═══\n' "${REMOTE}" "${REMOTE_DIR}"
git archive --format=tar HEAD | ssh "${REMOTE}" "mkdir -p ${REMOTE_DIR} && cd ${REMOTE_DIR} && tar xf - && mkdir -p logs ${RESULTS_ROOT} reports"
rsync -avz "${LOCAL_MANIFEST}" "${REMOTE}:${REMOTE_DIR}/logs/"

printf '═══ Submitting forecasting rerun array (%s tasks, throttle %s) ═══\n' "${TASKS}" "${THROTTLE}"
ARRAY_JOB=$(ssh "${REMOTE}" "cd ${REMOTE_DIR} && sbatch --parsable --job-name=forecast-mase-rerun --array=${ARRAY_SPEC} --export=ALL,RESULTS_ROOT=${RESULTS_ROOT} scripts/slurm_forecasting_mase_rerun_array.sh ${REMOTE_MANIFEST}")
printf 'Array job: %s\n' "${ARRAY_JOB}"

printf '═══ Submitting dependent postprocess job ═══\n'
POST_JOB=$(ssh "${REMOTE}" "cd ${REMOTE_DIR} && sbatch --parsable --dependency=afterok:${ARRAY_JOB} --export=ALL,RESULTS_ROOT=${RESULTS_ROOT} scripts/slurm_forecasting_mase_postprocess.sh")
printf 'Postprocess job: %s\n' "${POST_JOB}"

SUBMISSION_LOG="logs/${CAMPAIGN_ID}_submission.txt"
{
    echo "campaign_id=${CAMPAIGN_ID}"
    echo "commit=$(git rev-parse HEAD)"
    echo "manifest=${LOCAL_MANIFEST}"
    echo "tasks=${TASKS}"
    echo "throttle=${THROTTLE}"
    echo "results_root=${RESULTS_ROOT}"
    echo "array_job=${ARRAY_JOB}"
    echo "postprocess_job=${POST_JOB}"
    echo "remote=${REMOTE}"
    echo "remote_dir=${REMOTE_DIR}"
} > "${SUBMISSION_LOG}"

printf '\nSubmission log: %s\n' "${SUBMISSION_LOG}"
printf 'Monitor: ssh %s "squeue -u \\$USER"\n' "${REMOTE}"
printf 'Tail:    ssh %s "tail -f %s/logs/forecasting_mase_%s_*.out"\n' "${REMOTE}" "${REMOTE_DIR}" "${ARRAY_JOB}"
printf 'Fetch:   rsync -avz %s:%s/%s/ ./results/post_fix/ && rsync -avz %s:%s/reports/ ./reports/\n' \
    "${REMOTE}" "${REMOTE_DIR}" "${RESULTS_ROOT}" "${REMOTE}" "${REMOTE_DIR}"
