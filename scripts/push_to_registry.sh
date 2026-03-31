#!/usr/bin/env bash
# push_to_registry.sh — Push offshore-dl Docker image to GitHub Container Registry
#
# Prerequisites (one-time):
#   1. Create a GitHub Personal Access Token (PAT) with write:packages scope
#   2. Authenticate:
#      echo "$GITHUB_PAT" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
#
# Usage:
#   scripts/push_to_registry.sh                          # push with defaults
#   GITHUB_USER=myuser scripts/push_to_registry.sh       # override GitHub username
#   IMAGE_TAG=latest scripts/push_to_registry.sh         # override tag
#
# Environment variables:
#   GITHUB_USER  — GitHub username (required; no default)
#   IMAGE_TAG    — Docker image tag to push (default: train)
#   GHCR_REPO   — Full GHCR repository path (default: ghcr.io/$GITHUB_USER/offshore-dl)

set -euo pipefail

# --- Configuration -----------------------------------------------------------
if [[ -z "${GITHUB_USER:-}" ]]; then
    echo "✗ Error: GITHUB_USER is not set. Export it or pass inline:" >&2
    echo "    GITHUB_USER=myuser scripts/push_to_registry.sh" >&2
    exit 1
fi

IMAGE_TAG="${IMAGE_TAG:-train}"
GHCR_REPO="${GHCR_REPO:-ghcr.io/${GITHUB_USER}/offshore-dl}"
LOCAL_IMAGE="offshore-dl:${IMAGE_TAG}"
REMOTE_IMAGE="${GHCR_REPO}:${IMAGE_TAG}"

# --- Tag & Push ---------------------------------------------------------------
echo "▸ Tagging: ${LOCAL_IMAGE} → ${REMOTE_IMAGE}"
docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"

echo "▸ Pushing: ${REMOTE_IMAGE}"
docker push "${REMOTE_IMAGE}"

echo "✓ Push complete: ${REMOTE_IMAGE}"
echo ""
echo "To pull on NACAD via Singularity:"
echo "  singularity pull docker://${REMOTE_IMAGE}"

# --- Fallback: Air-gapped transfer -------------------------------------------
# If NACAD has no internet access or GHCR is blocked, use docker save instead:
#
#   # On workstation:
#   docker save offshore-dl:train | gzip > offshore-dl_train.tar.gz
#   scp offshore-dl_train.tar.gz nacad:/scratch/$USER/
#
#   # On NACAD:
#   singularity build offshore-dl_train.sif docker-archive:///scratch/$USER/offshore-dl_train.tar.gz
