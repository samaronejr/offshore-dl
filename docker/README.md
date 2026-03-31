# docker/

Container setup for running experiments with GPU support.

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: CUDA 12.4 + Python 3.11 + all dependencies including FM extras |
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

For HPC clusters without Docker, use Singularity/Apptainer — see `scripts/singularity_run.sh`.
