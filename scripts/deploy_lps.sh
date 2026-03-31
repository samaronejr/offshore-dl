#!/bin/bash
# Deploy to LPS cluster and submit all 3 experiments in parallel
set -euo pipefail

REMOTE="LPS_loginServer"
REMOTE_DIR="~/offshore-dl"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log() { echo "$(date '+%H:%M:%S') [deploy] $*"; }

log "=== Step 1: Create remote directory structure ==="
ssh $REMOTE "mkdir -p $REMOTE_DIR/{data/raw,results,reports,logs,mlruns}"

log "=== Step 2: Rsync project (code + configs + scripts) ==="
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.gsd/worktrees' \
    --exclude='mlruns' \
    --exclude='data' \
    --exclude='results' \
    --exclude='*.sif' \
    --exclude='*.tar.gz' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

log "=== Step 3: Rsync data ==="
rsync -avz --progress \
    "$LOCAL_DIR/data/raw/" "$REMOTE:$REMOTE_DIR/data/raw/"

log "=== Step 4: Transfer Docker image ==="
if [ -f /tmp/offshore-dl-train.tar.gz ]; then
    rsync -avz --progress /tmp/offshore-dl-train.tar.gz "$REMOTE:$REMOTE_DIR/"
    log "=== Step 5: Convert to Singularity (on GPU node) ==="
    ssh $REMOTE "cd $REMOTE_DIR && srun -p gpu -N1 --time=01:00:00 \
        singularity build offshore-dl-train.sif docker-archive://offshore-dl-train.tar.gz && \
        rm offshore-dl-train.tar.gz"
else
    log "WARNING: /tmp/offshore-dl-train.tar.gz not found. Build it first with:"
    log "  docker save offshore-dl:train | gzip > /tmp/offshore-dl-train.tar.gz"
    exit 1
fi

log "=== Step 6: Submit jobs ==="
ssh $REMOTE "cd $REMOTE_DIR && \
    JOB_3W=\$(sbatch --parsable scripts/slurm_3w.sh) && echo \"3W submitted: \$JOB_3W\" && \
    JOB_CDF=\$(sbatch --parsable scripts/slurm_cdf.sh) && echo \"CDF submitted: \$JOB_CDF\" && \
    JOB_GAN=\$(sbatch --parsable scripts/slurm_ganymede.sh) && echo \"Ganymede submitted: \$JOB_GAN\" && \
    echo 'All 3 jobs submitted! Monitor with: squeue -u \$USER'"

log "=== DEPLOY COMPLETE ==="
log "Monitor: ssh $REMOTE 'squeue -u \$USER'"
log "Logs:    ssh $REMOTE 'tail -f $REMOTE_DIR/logs/*.out'"
log "Results: rsync -avz $REMOTE:$REMOTE_DIR/results/ $LOCAL_DIR/results/"
