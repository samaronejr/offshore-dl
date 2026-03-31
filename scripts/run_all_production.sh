#!/usr/bin/env bash
# Run all production experiments sequentially after audit fixes
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [run_all] $*"; }

log "=== Phase 1: 3W Feature Classification (4 models, 5-fold) ==="
scripts/docker_run.sh python scripts/run_production_3w_features.py --device cuda 2>&1 | tail -20
log "=== Phase 1 DONE ==="

log "=== Phase 2: CDF Anomaly Detection (6 models) ==="
scripts/docker_run.sh python scripts/run_production_cdf.py --device cuda 2>&1 | tail -20
log "=== Phase 2 DONE ==="

log "=== Phase 3: Ganymede Forecasting (6 models × 4 horizons) ==="
scripts/docker_run.sh python scripts/run_production_ganymede.py --device cuda 2>&1 | tail -20
log "=== Phase 3 DONE ==="

log "=== Phase 4: Statistical Tests ==="
scripts/docker_run.sh python scripts/run_statistical_tests.py 2>&1 | tail -20
log "=== Phase 4 DONE ==="

log "=== ALL PRODUCTION RUNS COMPLETE ==="
