# docker/

Container setup for running experiments with GPU support. The Docker image is useful for local smoke tests and single-node GPU runs; large 3W HPO/Stage 2 campaigns are normally launched through Slurm on the HPC cluster.

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: CUDA 12.4 + Python 3.11 + core/dev/foundation-model dependencies by target. |
| `docker-compose.yml` | `train` service for GPU experiments and `mlflow` service for tracking. |
| `.env.template` | Template for environment variables such as HuggingFace token and MLflow URI. |

## Usage

```bash
# Build training image
docker build -t offshore-dl:train -f docker/Dockerfile --target train .

# Run the train service with GPU access
docker compose -f docker/docker-compose.yml up train

# Or use the convenience wrapper
scripts/docker_run.sh python scripts/run_production_ganymede.py --device cuda
```

## Result and data mounts

The compose/wrapper setup is expected to mount project data and outputs into the container:

- `./data:/app/data` — external datasets and processed artifacts. Data are not committed.
- `./results:/app/results` — generated experiment outputs by validity epoch/campaign.
- `./mlruns:/app/mlruns` — local MLflow artifacts when using file-backed tracking.
- `${HF_HOME:-~/.cache/huggingface}` — HuggingFace model cache for foundation models.

The train service uses a large shared-memory allocation for PyTorch DataLoader workers. Reduce it only if the host cannot provide enough RAM.

## Foundation model dependencies

Foundation model wrappers (Chronos, TimesFM, TiRex, MOMENT, MANTIS) require optional extras and may require `HF_TOKEN` for gated model downloads.

```bash
pip install -e ".[fm]"
```

Or build an FM-enabled image target if available in the Dockerfile:

```bash
docker build -t offshore-dl:fm -f docker/Dockerfile --target fm .
```

## HPO and large-memory jobs

3W HPO and Stage 2 runs can require substantial GPU VRAM, CPU RAM, and wall time. Prefer Slurm arrays for production campaigns:

```bash
sbatch scripts/slurm_hpo_3w_array.sh
sbatch scripts/hpc_job_3w.slurm
```

The latest Stage 2 `hydra_rocket` retry attempted an impractical multi-terabyte CPU allocation and should not be relaunched from Docker without redesigning the feature/memory strategy. RF window variants (`window360_rf`, `window1440_rf`) are more practical but should still be run with explicit memory limits and campaign output directories.

## HPC clusters (Singularity / Apptainer)

For HPC clusters without Docker, convert or pull the image as a Singularity/Apptainer image:

```bash
# Convert Docker image to SIF
apptainer build offshore-dl.sif docker-daemon://offshore-dl:train

# Run on HPC with GPU passthrough
apptainer exec --nv offshore-dl.sif python scripts/run_production_3w_features.py --device cuda

# Or use the provided wrapper
scripts/singularity_run.sh python scripts/run_production_3w_features.py --device cuda
```

See `scripts/README.md` for current Slurm wrappers and campaign-validation commands.
