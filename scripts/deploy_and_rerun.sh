#!/bin/bash
set -euo pipefail

PROJECT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="LPS_loginServer"
REMOTE_DIR="~/offshore-dl"
IMAGE_NAME="offshore-dl:train"
TAR_PATH="/tmp/offshore-dl-train.tar.gz"

echo "=== Step 1: Build Docker image ==="
docker build -t $IMAGE_NAME -f docker/Dockerfile .

echo "=== Step 2: Export image ==="
docker save $IMAGE_NAME | gzip > $TAR_PATH
echo "Image saved to $TAR_PATH ($(du -sh $TAR_PATH | cut -f1))"

echo "=== Step 3: Rsync code to LPS ==="
rsync -avP --exclude='.git' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.venv' \
    --exclude='data/' --exclude='results/' \
    --exclude='*.sif' --exclude='*.tar.gz' \
    $PROJECT_LOCAL/ $REMOTE:$REMOTE_DIR/

echo "=== Step 4: Transfer Docker image ==="
rsync -avP $TAR_PATH $REMOTE:$REMOTE_DIR/

echo "=== Step 5: Convert to Singularity on cluster ==="
ssh $REMOTE "cd $REMOTE_DIR && \
    mkdir -p logs && \
    srun --partition=gpu --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:00:00 \
    bash -c 'export SINGULARITY_CACHEDIR=/tmp/singularity_cache_\$\$ && mkdir -p \$SINGULARITY_CACHEDIR && singularity build --force offshore-dl-train.sif docker-archive://offshore-dl-train.tar.gz && rm -rf \$SINGULARITY_CACHEDIR'"

echo "=== Step 6: Submit all rerun jobs ==="
ssh $REMOTE "cd $REMOTE_DIR && \
    mkdir -p logs && \
    sbatch scripts/slurm_rerun_ganymede.sh && \
    sbatch scripts/slurm_rerun_3w_convtimenet.sh && \
    sbatch scripts/slurm_rerun_cdf.sh && \
    sbatch scripts/slurm_rerun_spe_berg.sh && \
    sbatch scripts/slurm_rerun_inner_mongolia.sh && \
    squeue -u \$USER"

echo ""
echo "=== All jobs submitted. Monitor with: ==="
echo "  ssh $REMOTE 'squeue -u \$USER'"
echo "  ssh $REMOTE 'tail -f $REMOTE_DIR/logs/rerun_ganymede_*.out'"
echo ""
echo "=== When complete, rsync results back: ==="
echo "  rsync -avz $REMOTE:$REMOTE_DIR/results/ $PROJECT_LOCAL/results/"
