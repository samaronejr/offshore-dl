#!/usr/bin/env bash
# Run all production experiments with MLflow tracking enabled.
#
# Prerequisites:
#   docker compose -f docker/docker-compose.yml up mlflow -d
#
# This logs all metrics, params, and artifacts to the MLflow server
# for reproducibility and visualization.
#
# Usage:
#   scripts/run_mlflow_tracked.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') [mlflow-run] $*"; }

# ── Start MLflow server ──
log "Starting MLflow server..."
docker compose -f docker/docker-compose.yml up mlflow -d
sleep 5

# Verify MLflow is up
if curl -s http://localhost:5000/health > /dev/null 2>&1 || curl -s http://localhost:5000 > /dev/null 2>&1; then
    log "MLflow server running at http://localhost:5000"
else
    log "WARNING: MLflow may not be ready yet, proceeding anyway..."
fi

# ── Run experiments with tracking ──
log "=== Phase 1: 3W Feature Classification (with MLflow) ==="
docker compose -f docker/docker-compose.yml run --rm train \
    python scripts/run_production_3w_features.py --device cuda --use-mlflow 2>&1 | tail -20
log "=== Phase 1 DONE ==="

log "=== Phase 2: Ganymede Forecasting (with MLflow) ==="
docker compose -f docker/docker-compose.yml run --rm train \
    python scripts/run_production_ganymede.py --device cuda --use-mlflow 2>&1 | tail -20
log "=== Phase 2 DONE ==="

log "=== ALL MLFLOW TRACKED RUNS COMPLETE ==="
log "View results: http://localhost:5000"
log "To stop MLflow: docker compose -f docker/docker-compose.yml down"
