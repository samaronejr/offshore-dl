# Baseline Results Report — Deep Learning Architectures for Offshore Production Monitoring

> **Status**: Complete nested evaluation (pre-Optuna HPO). All results use manually-tuned hyperparameters.
>
> **Date**: March 2026 (updated)

---

## 1. Experimental Setup

### 1.1 Datasets

| Property | 3W (Petrobras) | Ganymede (NSTA) | CDF (Cognite) |
|----------|:-:|:-:|:-:|
| **Task** | Fault Classification | Gas Production Forecasting | Anomaly Detection |
| **Samples** | 208,973 windows | 48,523 (raw) / 31,359 (filtered) | ~4,300 windows |
| **Features** | 27 sensors | 63 (9 raw + EMA lags) | 11 sensors |
| **Classes / Horizon** | 10 fault classes | h ∈ {7, 14, 30, 90} days | Reconstruction |
| **Input shape** | (14, 27) feature matrix | (90, 63) time series | (48, 11) time series |

### 1.2 Models

| Model | Type | Architecture | Params (clf/fcst) |
|-------|------|-------------|:----------:|
| **LSTM** | Trained | Bidirectional, 2 layers, hidden=256, attention pooling, LayerNorm | 2.30M / 2.24M |
| **DeepONet** | Trained | Flat MLP branch (clf) / CNN branch (fcst), rank=128 | 209K / 210K |
| **PatchTST** | Trained | Transformer, d_model=256, 8 heads, 3 layers, patch_len=7 | 433K / 402K |
| **Chronos** | FM (zero-shot) | amazon/chronos-t5-tiny, per-channel probabilistic forecasting | 8M (frozen) |
| **TimesFM** | FM (zero-shot) | Google TimesFM 1.0, univariate per-channel | 200M (frozen) |
| **TiRex** | FM (zero-shot) | xLSTM embeddings (6144-dim) + Random Forest (500 trees) | ~150M (frozen) |

### 1.3 Evaluation Protocol

**Nested evaluation** to prevent information leakage:
1. **Outer split**: 80/20 holdout (stratified-group for 3W, temporal for Ganymede/CDF)
2. **Inner CV**: K-fold within training pool (5-fold StratifiedGroupKFold for 3W, 3-fold ExpandingWindowCV for Ganymede, 3-fold SlidingWindowCV for CDF)
3. **Retrain**: Fresh model on full training pool with best-epoch early stopping
4. **Test**: Evaluate on held-out test set (never seen during training or selection)

---

## 2. 3W Fault Classification Results

### 2.1 Held-out Test Set (41,515 samples, 307 groups)

| Model | Accuracy | Macro F1 | AUC-PR | Inner CV Acc |
|-------|:--------:|:--------:|:------:|:------------:|
| **DeepONet** | **96.81%** | **0.9634** | 0.9328 | 96.36% ± 0.79 |
| **PatchTST** | 96.70% | 0.9634 | 0.9325 | 96.29% ± 1.22 |
| **LSTM** | 96.70% | 0.9625 | 0.9308 | 95.24% ± 1.22 |
| **TiRex** | 91.09% | 0.8938 | **0.9363** | 89.72% ± 2.06 |

### 2.2 Statistical Significance

Friedman test: χ² = 13.56, **p = 0.004** (significant).

Nemenyi post-hoc (CD = 2.10 at α = 0.05):
- **DeepONet and PatchTST significantly better than TiRex** on accuracy/F1
- Three trained models not significantly different from each other
- **TiRex significantly better than LSTM on AUC-PR** (TiRex rank 1.0 vs LSTM 4.0)

### 2.3 Per-Class Challenges

Class 2 (incipient BSW increase, 1.0% of data) is hardest for all models. TiRex recall drops to 67.3% on class 2 and 72.5% on class 4.

---

## 3. Ganymede Gas Production Forecasting

### 3.1 Held-out Test Set (multi-well)

| Model | h=7 R²_prod | h=14 | h=30 | h=90 | h=7 MAE |
|-------|:-----------:|:----:|:----:|:----:|:-------:|
| **TimesFM** | **0.826** | 0.763 | 0.656 | 0.413 | 0.740 |
| **TiRex** | 0.822 | **0.775** | **0.699** | **0.565** | **0.658** |
| **Chronos** | 0.734 | 0.655 | 0.576 | 0.372 | 0.741 |
| PatchTST | 0.590 | 0.403 | 0.233 | −0.417 | 1.776 |
| LSTM | −0.032 | −0.061 | 0.487 | 0.066 | 2.052 |
| DeepONet | −5.56 | −5.39 | −23.2 | −14.7 | 4.950 |

**Key finding**: All three foundation models massively outperform all trained models at every forecast horizon. TiRex degrades most gracefully with increasing horizon.

### 3.2 Statistical Significance

Friedman test on MAE: significant at h=7 (p=0.014), h=14 (p=0.035), h=30 (p=0.019). R²_prod not significant (limited power with 3 inner CV folds).

---

## 4. CDF Anomaly Detection

### 4.1 Held-out Test Set (864 samples, temporal 20%)

| Model | Error Mean | Error P50 | Error P95 |
|-------|:----------:|:---------:|:---------:|
| **LSTM** | **0.0041** | **0.0038** | 0.0057 |
| **PatchTST** | 0.069 | 0.053 | — |
| **DeepONet** | 0.217 | 0.185 | — |
| TiRex (train mean) | 130.9 | 40.7 | 802.7 |

> **Note**: CDF FM results (Chronos, TimesFM) pending Docker run. TiRex uses naive train-mean baseline.

---

## 5. Key Findings

1. **Statistical features enable efficient classification**: Compressing 720-step windows into 14-descriptor feature matrices yields >96.7% accuracy with all three trained models, reducing training time by ~50×.

2. **DeepONet leads classification with fewest parameters**: The operator-learning framework achieves the best accuracy (96.81%) with only 209K parameters — 11× fewer than LSTM.

3. **Foundation models dominate forecasting**: TimesFM, TiRex, and Chronos all achieve R²_prod > 0.7 at h=7 without any task-specific training, exceeding the best trained model (PatchTST: 0.59) by 24-40%.

4. **TiRex is the most versatile FM**: Achieves 91.1% classification (highest AUC-PR at 0.936) and the best long-horizon forecasting (R²_prod = 0.57 at h=90).

5. **Normalization is critical for CDF**: A bug in anomaly target normalization caused trained model errors of ~3,300. With proper per-sensor normalization, LSTM achieves error_mean = 0.004.

6. **DeepONet fails at forecasting**: Despite strong classification performance, DeepONet produces R² < -5 at all horizons — operator learning is not suited for autoregressive production forecasting.

---

## 6. Next Steps

- **Optuna HPO**: Search spaces defined in YAML configs; production script ready. Could shift trained model rankings.
- **CDF FM completion**: Chronos and TimesFM on CDF with nested protocol.
- **Model interpretability**: Attention visualization, SHAP values for feature importance.
