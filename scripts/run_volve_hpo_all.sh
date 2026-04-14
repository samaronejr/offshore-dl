#!/bin/bash
# Run all 16 Volve HPO studies sequentially
# Usage: bash scripts/run_volve_hpo_all.sh [logfile]
# Output: results/hpo/volve/{model}_{horizon}.json for all 16 combos

set -euo pipefail

LOGFILE="${1:-logs/volve_hpo_all.log}"
mkdir -p "$(dirname "$LOGFILE")"
mkdir -p results/hpo/volve

export MLFLOW_TRACKING_URI=mlruns

MODELS=(lstm deeponet patchtst tcn)
HORIZONS=(h7 h14 h30 h90)
N_TRIALS=30
DEVICE=cuda

total=0
done_count=0
failed_count=0
FAILED_COMBOS=()

for model in "${MODELS[@]}"; do
  for h in "${HORIZONS[@]}"; do
    total=$((total + 1))
    out="results/hpo/volve/${model}_${h}.json"
    
    # If file already exists with valid content, skip
    if [ -f "$out" ] && python -c "
import json, sys
try:
    d = json.load(open('$out'))
    assert 'best_value' in d.get('hpo', {}), 'missing best_value'
    assert d['hpo'].get('best_params'), 'empty best_params'
    sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null; then
      echo "$(date '+%Y-%m-%d %H:%M:%S') [SKIP] $model $h — valid JSON already exists" | tee -a "$LOGFILE"
      done_count=$((done_count + 1))
      continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') [START] $model $h (trial $total/16)" | tee -a "$LOGFILE"
    start_ts=$(date +%s)
    
    if MLFLOW_TRACKING_URI=mlruns python scripts/run_optuna_hpo.py \
        --dataset volve \
        --models "$model" \
        --horizon "$h" \
        --n-trials $N_TRIALS \
        --device $DEVICE \
        >> "$LOGFILE" 2>&1; then
      end_ts=$(date +%s)
      elapsed=$((end_ts - start_ts))
      echo "$(date '+%Y-%m-%d %H:%M:%S') [DONE] $model $h in ${elapsed}s" | tee -a "$LOGFILE"
      done_count=$((done_count + 1))
    else
      end_ts=$(date +%s)
      elapsed=$((end_ts - start_ts))
      echo "$(date '+%Y-%m-%d %H:%M:%S') [FAIL] $model $h after ${elapsed}s — retrying with --device cpu" | tee -a "$LOGFILE"
      FAILED_COMBOS+=("${model}_${h}")
      failed_count=$((failed_count + 1))
    fi
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') [PROGRESS] done=$done_count failed=$failed_count remaining=$((16 - done_count - failed_count))" | tee -a "$LOGFILE"
  done
done

echo "" | tee -a "$LOGFILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') === ALL RUNS COMPLETE ===" | tee -a "$LOGFILE"
echo "$(date '+%Y-%m-%d %H:%M:%S') done=$done_count failed=$failed_count" | tee -a "$LOGFILE"

# Final count
count=$(ls results/hpo/volve/*.json 2>/dev/null | wc -l)
echo "$(date '+%Y-%m-%d %H:%M:%S') Total JSON files: $count" | tee -a "$LOGFILE"

if [ "$failed_count" -gt 0 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED: ${FAILED_COMBOS[*]}" | tee -a "$LOGFILE"
  exit 1
fi

if [ "$count" -ge 16 ]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') PASS: $count files present" | tee -a "$LOGFILE"
  exit 0
else
  echo "$(date '+%Y-%m-%d %H:%M:%S') FAIL: only $count of 16 files present" | tee -a "$LOGFILE"
  exit 1
fi
