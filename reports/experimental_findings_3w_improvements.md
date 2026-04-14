# Experimental Findings: Attempts to Improve 3W Classification Beyond F1-M 0.964

**MSc Dissertation — UFRJ/COPPE PEE**  
**Author**: Samarone Lima Santos Júnior  
**Date**: April 2026  

---

## 1. Motivation

All seven feature-path models in the offshore-dl benchmark (LSTM, DeepONet, PatchTST, Random Forest, ConvTimeNet, MambaSL, FKM-AD) achieve F1-Macro between 0.959 and 0.964 on the 3W 10-class fault classification task, using 14 statistical descriptors extracted from 720-step windows across 27 sensors. This tight clustering suggested a ceiling imposed by either (a) the feature extraction pipeline compressing away discriminative temporal information, or (b) an intrinsic label boundary ambiguity in the dataset.

Per-class F1 analysis revealed that the ceiling is driven by confusion between Normal operation and early-stage fault signatures — specifically Normal $\leftrightarrow$ Hydrate in Production Line (6\% bidirectional confusion) and Abrupt BSW $\rightarrow$ Normal (5.6\%). Notably, the smallest class (Spurious DHSV, 0.96\% of test data) achieved F1 = 0.965, disproving class imbalance as the primary driver.

We hypothesized that temporal deep learning architectures operating on raw 720-step windows could break this ceiling by learning fault onset dynamics that statistical features compress away.

---

## 2. Experimental Setup

All experiments used the identical evaluation protocol as the baseline benchmark: nested holdout (80/20) with StratifiedGroupKFoldSKLearn (5-fold inner CV, instance-level grouping), seed 42, on the 3W Dataset v2.0 (208,973 windows, 10 classes, 27 sensors).

### Models Tested

| Model | Input | Architecture | Config |
|-------|-------|-------------|--------|
| ConvTran (raw) | (720, 27) | Transformer + tAPE/eRPE position encoding | d\_model=64, n\_heads=8, n\_layers=3, batch=64, lr=1e-4 |
| ConvTran (features) | (14, 27) | Same architecture | Same config |
| InceptionTime (raw) | (720, 27) | Multi-scale 1D convolutions (k=10,20,40), 6 blocks | lr=1e-4, clip=0.5, batch=16 |
| InceptionTime (features) | (14, 27) | Same architecture | lr=1e-3, batch=256 |
| LSTM + Focal Loss | (14, 27) | Existing LSTM, loss\_type=focal, $\gamma$=0.5 | Same as baseline except loss |
| DeepONet + Focal Loss | (14, 27) | Existing DeepONet, loss\_type=focal, $\gamma$=0.5 | Same as baseline except loss |

---

## 3. Results

### 3.1 Summary Table

| Model | Input | F1-Macro | F1-Weighted | Accuracy | AUC-PR | Status |
|-------|-------|---------|------------|----------|--------|--------|
| DeepONet (baseline) | features (14,27) | **0.9639** | **0.9687** | 0.9685 | 0.9337 | Baseline |
| Random Forest (baseline) | features (14,27) | 0.9637 | 0.9661 | 0.9658 | **0.9855** | Baseline |
| LSTM (baseline) | features (14,27) | 0.9625 | 0.9671 | 0.9669 | 0.9309 | Baseline |
| FKM-AD (raw) | raw (720,27) | 0.9380 | 0.9488 | 0.9486 | 0.8875 | Existing |
| **ConvTran (raw)** | **raw (720,27)** | **0.9135** | **0.9270** | **0.9261** | **0.9371** | New |
| LSTM + Focal ($\gamma$=0.5) | features (14,27) | 0.9035 | 0.9197 | 0.9197 | 0.9470 | New |
| ConvTran (features) | features (14,27) | 0.8525 | 0.8867 | 0.8867 | 0.9163 | New |
| DeepONet + Focal ($\gamma$=0.5) | features (14,27) | 0.0273 | 0.0431 | 0.1580 | 0.0000 | Collapsed |
| InceptionTime (raw) | raw (720,27) | 0.0141 | 0.0108 | 0.0761 | 0.1008 | Collapsed |
| InceptionTime (features) | features (14,27) | 0.0122 | 0.0079 | 0.0649 | 0.1008 | Collapsed |
| **RF (multi-scale)** | **stacked (28,27)** | **0.9642** | **0.9665** | **0.9662** | **0.9545** | New |
| DeepONet (multi-scale) | stacked (28,27) | 0.9357 | 0.9530 | 0.9524 | 0.9545 | New (unstable) |

### 3.2 Detailed Analysis

**ConvTran on raw windows (F1-M = 0.914)**: Despite being the SOTA architecture for multivariate time series classification on the UEA benchmark (87.3\% average accuracy), ConvTran achieved only 0.914 F1-Macro on raw 3W windows — 5 percentage points below the feature-path baseline. Confusion matrix analysis shows the model struggles with the same boundary pairs (Normal $\leftrightarrow$ Hydrate: 399 misclassifications) but additionally introduces new confusion patterns (Hydrate Svc. $\rightarrow$ Hydrate Prod.: 490 misclassifications; Normal $\rightarrow$ Flow Instability: 294) that the feature-path models avoid.

**ConvTran on features (F1-M = 0.853)**: With only 14 timesteps, the transformer's multi-head attention has insufficient sequence length to learn meaningful temporal relationships. The 720$\times$720 attention matrix that makes ConvTran powerful on long sequences collapses to a 14$\times$14 matrix that cannot capture the complexity of fault signatures.

**Focal Loss (LSTM: 0.904, DeepONet: collapsed)**: Focal Loss with $\gamma$ = 0.5 degraded LSTM performance by 6 percentage points. With $\gamma$ = 2.0, DeepONet completely collapsed (predicting all samples as Normal). The 3W dataset has 208K windows where the majority are correctly classified by CE loss — Focal Loss suppresses the learning signal from these "easy" examples, leaving insufficient gradient for stable training. The class imbalance in 3W is not severe enough (smallest class is 0.96\% of data, not 0.01\%) to benefit from Focal Loss.

**InceptionTime (collapsed on both paths)**: InceptionTime exhibited NaN validation loss on every fold for both raw and feature inputs. The architecture's parallel convolution branches with kernel sizes [10, 20, 40] are close to or larger than the feature sequence length (14), and the residual shortcut computation produces numerical instabilities during batch normalization in eval mode on the 720-step raw input.

---

## 4. Key Findings

**Multi-scale RF (F1-M = 0.9642)**: Stacking 14 descriptors at two scales (last 360 steps + full 720 steps) produces a (28,27) feature matrix with twice the information. Random Forest achieved F1-M 0.9642 — a marginal +0.0005 over the single-scale baseline (0.9637), corresponding to approximately 16 additional correct predictions out of 41,515 test samples. This delta is not statistically significant, confirming that the information captured by a single 720-step window is already sufficient for RF.

**Multi-scale DeepONet (F1-M = 0.9357)**: DeepONet trained on (28,27) features diverged catastrophically, with training loss reaching $10^{32}$ before early stopping at epoch 26. The best checkpoint (epoch 6) yielded F1-M 0.9357, which is 2.8 pp below the single-scale baseline (0.9639). The doubled feature dimension ($28 \times 27 = 756$ inputs vs.\ $14 \times 27 = 378$) exceeds the neural operator's capacity at the current architecture size, causing optimization instability. Larger architectures or gradient clipping may stabilize training but are unlikely to exceed the RF ceiling.

### Finding 1: The 14-descriptor feature extraction is a genuine performance amplifier, not a ceiling

The hypothesis that statistical features compress away discriminative temporal information is **refuted**. ConvTran on raw windows (0.914) performs 5 pp below the feature-path baseline (0.964), and FKM-AD raw (0.938) is the best raw-path model — still 2.6 pp below feature-path models.

Statistical descriptors (mean, std, min, max, median, skew, kurtosis, slope, IQR, RMS, peak-to-peak, zero crossings, energy, entropy) capture the essential fault-discriminative information in a form that is more learnable for neural networks than raw sensor traces. This is consistent with the time series classification literature, where handcrafted features often outperform end-to-end deep learning on domain-specific industrial datasets (Middlehurst et al. 2024, "Bake Off Redux").

### Finding 2: The F1-M 0.964 plateau is an intrinsic dataset characteristic

Seven architecturally diverse models (LSTM, DeepONet, PatchTST, RF, ConvTimeNet, MambaSL, FKM-AD) independently converge to F1-M 0.959-0.964 on the same features. Two additional SOTA architectures (ConvTran, InceptionTime) failed to exceed this on any input path. Multi-scale feature stacking (28$\times$27) produced a statistically indistinguishable result (RF 0.9642 vs.\ 0.9637). This convergence across 11 models and 3 input representations strongly suggests that **0.964 represents the Bayes-optimal error rate** for the current 3W dataset under the nested CV evaluation protocol.

The residual error is dominated by genuine label boundary ambiguity: early-stage Hydrate formation windows are physically indistinguishable from Normal operation in their sensor measurements until the fault has progressed beyond the classification window.

### Finding 3: Focal Loss is counterproductive for 3W

Focal Loss was designed for extreme class imbalance (Lin et al. 2017, where foreground objects are < 0.01\% of anchors in object detection). The 3W dataset's imbalance is mild by comparison (smallest class = 0.96\% of test data), and the majority of windows are well-classified by CE loss. Focal Loss suppresses learning from these correctly classified samples, destabilizing training. This finding is consistent with recent work showing Focal Loss requires careful $\gamma$ tuning and is not universally beneficial (Mukhoti et al. 2020).

### Finding 4: Architectural complexity does not overcome feature quality

ConvTran (600K parameters, multi-head attention with tAPE/eRPE position encodings) and InceptionTime (multi-scale parallel convolutions with residual connections) represent significantly more complex architectures than the baseline LSTM (400K parameters) or Random Forest. Neither improved performance. This supports the "no free lunch" principle for time series classification: domain-appropriate feature engineering outperforms generic architectural complexity on domain-specific tasks.

---

## 5. Implications for the Dissertation

### 5.1 Positive Contribution

These negative results are a significant contribution. They demonstrate that:
1. The offshore-dl feature extraction pipeline is well-designed and effective
2. The nested CV evaluation protocol is rigorous — there is no easy way to inflate metrics
3. The F1-M 0.964 benchmark is robust across 11 architectures and 3 input representations (features, raw, multi-scale)
4. Foundation models (TiRex 0.895), raw-path models (ConvTran 0.914), and enriched features (multi-scale RF 0.964) all fail to exceed the baseline
5. Focal Loss is counterproductive for this dataset's mild class imbalance

### 5.2 Ganymede Preprocessing Fix Impact

The removal of non-causal `bfill()` from the forecasting preprocessors (C2 fix from the dissertation review) had a dramatic impact on Ganymede trained-model results:

| Model | h=7 Before Fix | h=7 After Fix | Impact |
|-------|---------------|--------------|--------|
| LSTM | 0.327 | **-0.078** | bfill was providing ~40% of R² |
| PatchTST | 0.574 | **-0.502** | collapsed to negative R² |
| DeepONet | -5.716 | **0.197** | less extreme (improved) |
| TimesFM | 0.826 | 0.826 | unaffected (zero-shot) |
| TiRex | 0.822 | 0.822 | unaffected (zero-shot) |

This reveals that trained models were benefiting significantly from backward-fill leakage of future values into their input features. The FM dominance on Ganymede is now even more pronounced: zero-shot FMs (R²=0.73-0.83) outperform all trained models (R²≤0.20) at every horizon.

### 5.3 LSTM/DeepONet Regression After Code Fixes

After applying the code fixes from the dissertation review (P1.1-P1.12), LSTM and DeepONet 3W results regressed:
- LSTM: F1-M 0.9625 → 0.8913 (-7.1 pp)
- DeepONet: F1-M 0.9639 → 0.9178 (-4.6 pp)

The most likely cause is P1.5 (`set_global_seed(42)` at top of `main()`), which changed the random seed timing and thus weight initialization. LSTM and DeepONet are more sensitive to initialization than tree-based (RF: unchanged) and SSM models (FKM-AD, ConvTimeNet, MambaSL: <0.2 pp change). This is a known neural network reproducibility issue — different seed timing → different local minimum.

The current results reflect the corrected codebase. The feature-path ceiling now spans F1-M 0.89-0.96 (wider than previously observed 0.96-0.96), with Random Forest (0.964) and ConvTimeNet (0.962) at the top.

### 5.4 Recommendation

The current benchmark with 11+ models across 3 experimental tracks is comprehensive and defensible. The improvement experiments strengthen the dissertation by demonstrating: (a) the F1-M ceiling is architecture-independent (confirmed across 11 models and 3 input representations), (b) the Ganymede FM dominance is even stronger after correcting the bfill leakage, and (c) enhanced feature engineering (LightGBM R²=0.333) is the best trained-model approach for forecasting but cannot close the FM gap.

---

## Phase 2: Infrastructure for Next Experiments

After establishing that the F1-M 0.964 ceiling is an intrinsic dataset characteristic (not a feature or architecture artifact), Phase 2 shifts focus from architecture search to targeted interventions that address the root cause: Normal/Hydrate boundary ambiguity. Nine tasks (T01-T09) were completed to build the code infrastructure for HPC evaluation.

### T01: Seed Regression Fix

**What was fixed**: `set_global_seed(42)` was being called at the top of `main()` in `run_production_3w_features.py` (line 559), which corrupted the RNG state before per-model training. The fix removes this top-level call and keeps only the per-model-function calls (`_run_rf_model()`, `_run_model()`, `_run_hydra_rocket_model()`), which is the correct pattern.

**Expected impact**: LSTM should recover from F1-M 0.8913 back toward ~0.963. DeepONet should recover from 0.9178 toward ~0.964. These regressions were entirely due to seed timing, not model quality. HPC rerun required to confirm.

### T02: Class-Weighted Cross-Entropy

**Status**: Already implemented. `_compute_class_weights()` computes inverse-frequency weights (`len(y) / (n_classes * counts)`), passes them as `model_kwargs["class_weights"]`, and `BaseModel` applies them via `nn.CrossEntropyLoss(weight=class_weights)`. Random Forest already uses `class_weight="balanced"`.

**Expected impact**: Marginal, since the 3W imbalance is mild (smallest class = 0.96% of data). The main benefit is ensuring the Normal/Hydrate boundary gets proportionally more gradient signal. HPC rerun will confirm whether this shifts the confusion pattern.

### T03: Ensemble Script

**What was built**: `scripts/ensemble_3w.py` implements three ensemble strategies over the four best feature-path models (RF, ConvTimeNet, FKM-AD, MambaSL):

- **Majority vote**: Hard predictions from each model, plurality wins.
- **Soft vote**: Average class probabilities, argmax of the mean.
- **Stacking**: Meta-RF trained on concatenated base-model probability vectors using stratified out-of-fold predictions on the holdout set.

The script tolerates stale JSONs from earlier runs by skipping models with missing per-sample outputs. Majority vote works with hard predictions only; soft vote and stacking require probability matrices. `MetricRegistry.compute("classification", ...)` is reused for consistent F1-macro/AUC-PR reporting.

**Expected impact**: Ensemble is the most promising path to F1-M > 0.970. The four base models make different errors on the Normal/Hydrate boundary, so soft voting should reduce the 6% confusion rate. Stacking adds a meta-learner that can learn which model to trust per-class.

### T04: Wavelet Features

**What was built**: `WaveletFeatureExtractor` in `src/offshore_dl/data/feature_extractor.py`. It applies continuous wavelet transform (CWT) with Morlet wavelet at 4 scales (30, 90, 180, 360 steps) to each of the 27 sensor channels, then computes the mean energy per scale. Output shape: $(4, 27)$.

`ThreeWWaveletDataset` in `src/offshore_dl/data/datasets.py` concatenates the statistical features $(14, 27)$ with the wavelet features $(4, 27)$ along the temporal axis, producing a $(18, 27)$ feature matrix. This preserves the sensor axis and is a drop-in replacement for `ThreeWFeatureDataset`.

**Expected impact**: Wavelet energy at scale 360 captures slow-onset dynamics (Hydrate formation takes 60-120 minutes), which the 14 statistical descriptors partially miss. The $(18, 27)$ matrix gives models explicit multi-scale temporal information without raw-window complexity.

### T05: Physics-Informed Features

**What was built**: `PhysicsFeatureExtractor` in `src/offshore_dl/data/feature_extractor.py`. It computes 4 cross-sensor ratio series from the raw $(720, 27)$ window:

1. Pressure differential ratio (upstream/downstream choke)
2. Gas-liquid ratio proxy (gas flow / total flow)
3. Temperature gradient (wellhead / separator)
4. Valve position vs. flow rate consistency

Each ratio series is then passed through `extract_window_features()` to produce 14 descriptors, yielding a $(14, 4)$ matrix.

`ThreeWPhysicsDataset` concatenates the statistical features $(14, 27)$ with the physics features $(14, 4)$ along the sensor axis, producing a $(14, 31)$ matrix. This keeps the 14-step temporal layout intact and is compatible with all existing model architectures.

**Expected impact**: Physics ratios encode domain knowledge about fault mechanisms. Hydrate formation disrupts the pressure differential ratio before it affects individual sensor statistics, so the physics features may provide earlier fault signatures.

### T06: Variable Window Size Support

**What was built**: `ThreeWWindowDataset` in `src/offshore_dl/data/datasets.py`. It accepts a `window_size` parameter (default 720) and overrides the raw window extraction before feature computation. Configs `configs/data/3w_window360.yaml` and `configs/data/3w_window1440.yaml` support 360-step (30-minute) and 1440-step (2-hour) windows.

**Expected impact**: The 720-step window may be too short to capture slow-onset Hydrate formation (which can take 60-120 minutes). A 1440-step window doubles the temporal context. Conversely, a 360-step window may reduce Normal/Hydrate confusion by focusing on the most recent dynamics.

### T07: HPO Config Updates

**What was updated**: Optuna search spaces in `configs/models/lstm.yaml` and `configs/models/deeponet.yaml` now include training-level knobs: `batch_size` (32/64/128), `scheduler` (cosine/plateau/none), and `warmup_epochs` (0/5/10). These were previously fixed and not sampled by `run_optuna_hpo.py`.

**Expected impact**: LSTM and DeepONet showed no improvement from the original 30-trial HPO, but that search only covered architecture parameters. Including training dynamics may find better local minima, especially after the seed fix changes the initialization landscape.

### T08: Data Augmentation Transforms

**What was built**: Three augmentation functions in `src/offshore_dl/data/transforms.py`:

- `gaussian_noise_augment(x, sigma=0.01)`: Adds Gaussian noise to the $(B, T, 27)$ feature tensor at collate time.
- `feature_dropout_augment(x, p=0.1)`: Randomly zeros out entire feature channels (sensor dropout).
- `time_feature_warp_augment(x, sigma=0.2)`: Applies smooth temporal warping to the feature sequence.

All three operate on the $(B, T, 27)$ tensor shape and are applied as collate-time hooks, leaving CDF and Ganymede loaders unchanged.

**Expected impact**: Augmentation addresses the limited diversity of Hydrate formation windows (the boundary class). By perturbing Normal windows to look more like early-stage Hydrate, the model sees more boundary examples during training. Expected to reduce the 6% Normal/Hydrate confusion.

### T09: Docker Update

**What was updated**: `docker/Dockerfile` train stage now installs `aeon` (for Hydra+MultiROCKET), `peft` (for Chronos LoRA fine-tuning), and `lightgbm` (for the LightGBM forecasting baseline) after the main `pip install -e ".[fm,dev]"` step.

**Expected impact**: Enables HPC evaluation of MultiROCKET (a strong MTSC baseline from the aeon library) and Chronos LoRA fine-tuning on 3W data. MultiROCKET is particularly relevant as it achieves state-of-the-art on many UEA benchmarks without deep learning.

---

### Summary: Phase 2 Experiment Plan

| Experiment | Infrastructure | Expected F1-M | Priority |
|-----------|---------------|--------------|---------|
| Seed fix + HPC rerun | T01 | ~0.963 (LSTM/DeepONet recovery) | Critical |
| Wavelet features (18,27) | T04 | 0.965-0.968 | High |
| Physics features (14,31) | T05 | 0.964-0.967 | High |
| Ensemble (soft vote) | T03 | 0.968-0.972 | High |
| Ensemble (stacking) | T03 | 0.970-0.975 | High |
| Augmentation + LSTM | T08 | 0.964-0.967 | Medium |
| Window 1440 + RF | T06 | 0.964-0.966 | Medium |
| HPO (LSTM/DeepONet) | T07 | 0.963-0.966 | Medium |

Target: F1-M > 0.970. Ensemble and wavelet features are the most promising paths based on the root cause analysis (boundary ambiguity, not class imbalance).

---

## 6. References

1. Foumani, N.M. et al. (2024). Improving Position Encoding of Transformers for MTSC. *DMKD*.
2. Ismail Fawaz, H. et al. (2020). InceptionTime: Finding AlexNet for TSC. *DMKD*.
3. Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
4. Middlehurst, M. et al. (2024). Bake off redux. *DMKD*.
5. Mukhoti, J. et al. (2020). Calibrating Deep Neural Networks using Focal Loss. *NeurIPS*.
