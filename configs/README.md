# configs/

Hierarchical YAML configuration using [OmegaConf](https://omegaconf.readthedocs.io/).

Configuration is merged in order: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides.

## Structure

```
configs/
├── base.yaml       # Global defaults: seed, device, training params, MLflow, Optuna
├── data/            # Dataset-specific settings (sensors, classes, horizons, splits)
│   ├── 3w.yaml      #   Petrobras 3W (10-class classification, 27 sensors)
│   ├── ganymede.yaml #   Ganymede (gas forecasting, 7 wells, 4 horizons)
│   └── cdf.yaml      #   CDF (anomaly detection, 12 sensors)
└── models/          # Model architecture + Optuna search spaces
    ├── lstm.yaml
    ├── deeponet.yaml
    ├── patchtst.yaml
    ├── mlp.yaml
    └── xgboost.yaml
```

## Usage

```python
from offshore_dl.utils.config import load_config
cfg = load_config(model="lstm", dataset="ganymede")
# cfg.model.architecture.hidden_size → 256
# cfg.model.optuna_search_space.hidden_size → [64, 128, 256, 512]
```
