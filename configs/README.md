# configs/

Hierarchical YAML configuration using [OmegaConf](https://omegaconf.readthedocs.io/).

Configuration is merged in order: `base.yaml` ← `data/*.yaml` ← `models/*.yaml` ← CLI overrides.

## Structure

```
configs/
├── base.yaml                   # Global defaults: seed, device, training params, MLflow, Optuna
├── data/                       # Dataset-specific settings (sensors, classes, horizons, splits)
│   ├── 3w.yaml                 #   Petrobras 3W (10-class classification, 27 sensors)
│   ├── ganymede.yaml           #   Ganymede (gas forecasting, 7 wells, 4 horizons)
│   ├── cdf.yaml                #   CDF (anomaly detection, 12 sensors)
│   ├── spe_berg.yaml           #   SPE Berg (well control classification)
│   ├── volve.yaml              #   Volve (production forecasting)
│   ├── inner_mongolia.yaml     #   Inner Mongolia (fault detection)
│   ├── 3w_multiscale.yaml      #   3W with multi-scale window settings
│   ├── 3w_physics.yaml         #   3W with physics-informed constraints
│   └── 3w_wavelet.yaml         #   3W with wavelet feature augmentation
└── models/                     # Model architecture + Optuna search spaces
    ├── lstm.yaml
    ├── deeponet.yaml
    ├── patchtst.yaml
    ├── tcn.yaml
    ├── convtimenet.yaml
    ├── convtran.yaml
    ├── inception_time.yaml
    ├── mambasl.yaml
    ├── fkmad.yaml
    ├── fkmad_hpo.yaml
    ├── random_forest.yaml
    ├── lstm_focal.yaml
    └── deeponet_focal.yaml
```

## Usage

```python
from offshore_dl.utils.config import load_merged_config
cfg = load_merged_config(model="lstm", dataset="ganymede")
# cfg.model.architecture.hidden_size → 256
# cfg.model.optuna_search_space.hidden_size → [64, 128, 256, 512]
```
