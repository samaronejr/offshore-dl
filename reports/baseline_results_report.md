# Historical Baseline Results Report — Deep Learning Architectures for Offshore Production Monitoring

> **Status**: Historical March 2026 snapshot. This report is superseded by
> `reports/dissertation_results_2026-05.{tex,pdf}` and
> `reports/dissertation_result_manifest_2026-05.md`; do not cite it for current
> dissertation rankings.
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

The March 2026 Ganymede table used pre-fix forecasting semantics and is no
longer repeated here to prevent stale citation. Use the May 2026 dissertation
manifest for current Ganymede rows. Safe current wording is metric-specific:
zero-shot FMs lead by MAE/RMSE, while trained LSTM/TCN lead by grouped MASE.

### 3.2 Statistical Significance

Friedman test on MAE: significant at h=7 (p=0.014), h=14 (p=0.035), h=30 (p=0.019). R²_prod not significant (limited power with 3 inner CV folds).

---

## 4. CDF Anomaly Detection

### 4.1 Held-out Test Set (864 samples, temporal 20%)

The March 2026 CDF table is superseded by the strict-gap post-fix rerun in
`reports/cdf_post_fix_summary_2026-05.json`. Current CDF reporting separates
trained reconstruction metrics (`error_*`) from foundation forecast metrics
(`forecast_error_*`).

---

## 5. Key Findings

1. **Statistical features enable efficient classification**: Compressing 720-step windows into 14-descriptor feature matrices yields >96.7% accuracy with all three trained models, reducing training time by ~50×.

2. **DeepONet leads classification with fewest parameters**: The operator-learning framework achieves the best accuracy (96.81%) with only 209K parameters — 11× fewer than LSTM.

3. **Forecasting conclusions are superseded by May 2026 post-fix results**: Current wording must name the metric family instead of stating a universal winner.

4. **TiRex March snapshot claims are historical**: Use the current manifest before citing any cross-task versatility claim.

5. **CDF conclusions are superseded by the strict-gap rerun**: Use the May 2026 CDF summary and keep trained-vs-FM anomaly semantics separate.

6. **DeepONet fails at forecasting**: Despite strong classification performance, DeepONet produces R² < -5 at all horizons — operator learning is not suited for autoregressive production forecasting.

---

## 6. Next Steps

- **Optuna HPO**: Search spaces defined in YAML configs; production script ready. Could shift trained model rankings.
- **CDF FM completion**: superseded by the May 2026 post-fix CDF rerun.
- **Model interpretability**: Attention visualization, SHAP values for feature importance.
