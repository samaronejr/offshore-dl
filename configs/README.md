# configs/

Hierarchical YAML configuration using [OmegaConf](https://omegaconf.readthedocs.io/).

Configuration is merged in order: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides.

## Structure

```text
configs/
├── base.yaml                   # Global defaults: seed, device, training, MLflow, Optuna
├── data/                       # Dataset and campaign-variant settings
│   ├── 3w.yaml                 # Petrobras 3W, standard 720-window classification
│   ├── 3w_window_360.yaml      # 3W short-window Stage 2 RF variant
│   ├── 3w_window_1440.yaml     # 3W long-window Stage 2 RF variant
│   ├── 3w_multiscale.yaml      # 3W multiscale feature variant
│   ├── 3w_physics.yaml         # 3W physics-informed feature variant
│   ├── 3w_wavelet.yaml         # 3W wavelet feature variant
│   ├── ganymede.yaml           # Ganymede gas forecasting, 7 wells, 4 horizons
│   ├── cdf.yaml                # CDF anomaly detection, 12 sensors
│   ├── spe_berg.yaml           # SPE Berg forecasting
│   ├── volve.yaml              # Volve production forecasting
│   └── inner_mongolia.yaml     # Inner Mongolia forecasting
└── models/                     # Model architecture + Optuna search spaces
    ├── lstm.yaml
    ├── lstm_focal.yaml
    ├── deeponet.yaml
    ├── deeponet_focal.yaml
    ├── deeponet_recon_clf.yaml
    ├── deeponet_trunk_clf.yaml
    ├── patchtst.yaml
    ├── tcn.yaml
    ├── convtimenet.yaml
    ├── convtran.yaml
    ├── inception_time.yaml
    ├── mambasl.yaml
    ├── fkmad.yaml
    ├── fkmad_baseline.yaml
    ├── fkmad_hpo.yaml
    ├── random_forest.yaml
    ├── chronos.yaml
    ├── timesfm.yaml
    └── tirex.yaml
```

## Usage

```python
from offshore_dl.utils.config import load_merged_config
cfg = load_merged_config(model="lstm", dataset="ganymede")
# cfg.model.architecture.hidden_size → 256
# cfg.model.optuna_search_space.hidden_size → [64, 128, 256, 512]
```

CLI overrides use OmegaConf dotlist syntax:

```bash
python -m offshore_dl.run_experiment --model lstm --dataset ganymede training.batch_size=128
```

## Benchmark-specific guidance

- **3W Stage 1 HPO** uses the standard `3w.yaml` setup and macro-F1 as the primary objective. Promote only HPO outputs that pass `scripts/validate_hpo_3w_results.py`.
- **3W Stage 2** configs (`3w_window_*`, `3w_multiscale`, `3w_physics`, `3w_wavelet`) are campaign variants. Report them separately from the standard 720-window leaderboard.
- **Ganymede** forecasting configs define four horizons (`h7`, `h14`, `h30`, `h90`) and support multi-well/per-well evaluation. MAE/RMSE and MASE can rank models differently, so reports should name the metric.
- **CDF** needs post-fix reruns before current anomaly-detection claims because historical runs predate the strict raw-row CV-gap repair.

## Adding configs

- New dataset: add `configs/data/<dataset>.yaml`, document split/window/horizon semantics, and register the dataset in the experiment entry point or production script.
- New trained model: add `configs/models/<model>.yaml` with architecture parameters and an `optuna_search_space` block when HPO is supported.
- New one-off campaign variant: prefer a clearly named data config and document whether it is comparable to the main benchmark or must be reported separately.
