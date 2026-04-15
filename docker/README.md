# docker/

Container setup for running experiments with GPU support.

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: CUDA 12.4 + Python 3.11 + all dependencies including optional foundation model extras |
| `docker-compose.yml` | Two services: `train` (GPU training) and `mlflow` (experiment tracking server) |
| `.env.template` | Template for environment variables (HuggingFace token, MLflow URI) |

## Usage

```bash
# Build
docker build -t offshore-dl:train -f docker/Dockerfile --target train .

# Run with GPU
docker compose up train

# Or use the convenience wrapper
scripts/docker_run.sh python scripts/run_production_ganymede.py --device cuda
```

## Volume Mounts

The `docker-compose.yml` configures the following persistent volumes:

- `./results:/app/results` — experiment outputs
- `./optuna.db:/app/optuna.db` — Optuna HPO study database (persisted across runs)
- `${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface` — HuggingFace model cache for foundation models

## Foundation Model Dependencies

Foundation model wrappers (Chronos, TimesFM, TiRex, MOMENT, MANTIS) require optional extras. Install inside the container with:

```bash
pip install -e ".[fm]"
```

Or build the `fm` Docker target:

```bash
docker build -t offshore-dl:fm -f docker/Dockerfile --target fm .
```

## HPC Clusters (Singularity / Apptainer)

For HPC clusters without Docker, convert the image to Singularity/Apptainer format:

```bash
# Convert Docker image to SIF
apptainer build offshore-dl.sif docker://offshore-dl:train

# Run on HPC
apptainer exec --nv offshore-dl.sif python scripts/run_production_3w_features.py

# Or use the provided wrapper
scripts/singularity_run.sh python scripts/run_production_3w_features.py
```

See `scripts/slurm/` for ready-made Slurm job scripts.
