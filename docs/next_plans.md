# Next Steps Plan

**Date:** 2026-04-14
**Context:** Post-review, post-cleanup session. HPC jobs running, local HPO in progress.

---

## Phase A: Wait for Running Jobs (this week)

### A1. HPC Jobs (~3-4 days remaining)
- 25 jobs cycling through: SPE Berg, Volve, Inner Mongolia, Ganymede (with retrain fix), 3W features, stat tests
- All jobs include MASE denominator fix and retrain validation split fix
- Monitor: `ssh LPS_loginServer squeue -u $(whoami) -p gpu`

### A2. Local HPO (~16 hours remaining)
- SPE Berg PatchTST h14: trial 26/30, best MAE=2.668 (trial #9)
- Best params: d_model=512, n_heads=16, n_layers=3, lr=1.4e-4, dropout=0.063
- Running on local RTX 3090

### A3. Fetch Results When Complete
```bash
rsync -avz LPS_loginServer:~/offshore-dl/results/ ./results/
rsync -avz LPS_loginServer:~/offshore-dl/reports/ ./reports/
```

---

## Phase B: Analyze New Results (after HPC completes)

### B1. Check Retrain Fix Impact on Ganymede R2_prod
- Compare old vs new R2_prod for trained models (LSTM, DeepONet, PatchTST, TCN)
- Old: LSTM h30 R2_prod=-0.280 (with temporal-tail retrain split)
- Expected: improvement due to random retrain split, but may still be negative (genuine non-stationarity)
- **If still negative:** document as legitimate finding (FM advantage via local context conditioning)
- **If improved to positive:** the temporal-tail split was the primary cause

### B2. Model Diversity Analysis (Phase 0b from consensus plan)
- Compute pairwise Cohen's kappa across 3W models using stored predictions
- **If kappa < 0.95:** models disagree enough for ensemble to help -> proceed to Phase C
- **If kappa >= 0.95:** models agree on errors -> skip ensemble, go to Phase D
- Script: `scripts/ensemble_3w.py` (already exists, needs stored predictions from 3W re-run)
- Compute: 0 GPU-hours (pure analysis)

### B3. Verify MASE Values
- Check that new results have non-None y_train in MASE computation
- Compare old vs new MASE values across all forecasting datasets
- Update dissertation tables

---

## Phase C: Quick Wins — Ensembles (0-2 GPU-hours)

*Only pursue if B2 shows model diversity (kappa < 0.95)*

### C1. 3W Ensemble
- Models: RF, ConvTimeNet, FKMAD, MambaSL, PatchTST (top 5 by F1)
- Strategies (try in order, stop if target met):
  1. Majority vote
  2. Soft vote (uncalibrated)
  3. Weighted soft vote
  4. RF stacking with stratified OOF
- Target: F1 > 0.964 (beat RF)
- Script: `scripts/ensemble_3w.py`

### C2. Ganymede FM Ensemble
- Models: TiRex, Chronos, TimesFM (if R2_prod remains positive)
- Strategy: weighted average by per-fold MAE
- Compute: ~1 GPU-hour for FM inference

---

## Phase D: Targeted Improvements (5-20 GPU-hours)

### D1. Label Smoothing (zero extra compute)
- Add `label_smoothing=0.05` to model kwargs for top 3W models
- Models: ConvTimeNet, FKMAD, PatchTST
- Config change only, re-run via HPC: `sbatch scripts/slurm_run_model.sh 3w_features <model>`

### D2. SupCon Pre-Training (highest expected impact)
- Target: Normal-Hydrate confusion (main error source)
- Module: `src/offshore_dl/training/supcon.py` (already implemented)
- Usage:
  ```python
  from offshore_dl.training.supcon import SupConPreTrainer
  encoder_state = pretrainer.pretrain(dataset, train_indices, epochs=50)
  # Load encoder_state into classification model, then fine-tune
  ```
- Expected: 1-3% F1 lift on confused classes
- Compute: ~5-10 GPU-hours (pre-training + fine-tuning)

### D3. Multi-Window-Scale Experiment
- Literature shows 120-second windows optimal for hydrate detection (Lopes et al. 2024)
- Current: 720 timesteps (12 min)
- Test: 60/120/360/720 windows with ConvTimeNet and FKMAD
- Compute: ~4 GPU-hours per window size x 2 models = ~16 GPU-hours
- May need to adjust feature extractor for different window sizes

### D4. Apply HPO Best Params
- Apply SPE Berg PatchTST h14 best params (from local HPO) to production sweep
- Update `configs/models/patchtst.yaml` with: d_model=512, n_heads=16, n_layers=3, lr=1.4e-4, dropout=0.063
- Re-run SPE Berg PatchTST via HPC

---

## Phase E: New FM Baselines (10-15 GPU-hours)

### E1. MOMENT (Fine-tuned FM)
- Module: `src/offshore_dl/models/moment_wrapper.py` (already implemented)
- Install: `pip install momentfm` (add to Dockerfile)
- Strategy: 3-stage fine-tuning (linear probe -> LoRA r=4 -> optional full)
- Dataset: 3W classification (primary), Ganymede forecasting (secondary)
- Expected: stronger FM baseline than zero-shot Chronos/TimesFM/TiRex

### E2. Mantis (Classification-Specific FM)
- Module: `src/offshore_dl/models/mantis_wrapper.py` (already implemented)
- Install: `pip install mantis-fm` (or from github.com/vfeofanov/mantis)
- Strategy: frozen encoder + RF on embeddings (mirrors TiRex pipeline)
- Dataset: 3W classification
- Expected: direct comparison to TiRex approach with a classification-specific encoder

### E3. Rebuild Docker Image
- Add `momentfm` and `mantis-fm` to `pyproject.toml` optional deps
- Rebuild: `docker build -t offshore-dl:train -f docker/Dockerfile --target train .`
- Convert to SIF and deploy to HPC

---

## Phase F: Statistical Analysis & Dissertation (after all experiments)

### F1. Regenerate Statistical Tests
```bash
python scripts/run_statistical_tests.py
```
- Now includes Holm correction on Wilcoxon and proper tie handling in Nemenyi

### F2. Update Dissertation Tables
- New MASE values (corrected denominator)
- New R2_prod values (retrain fix)
- Any improvements from ensemble/SupCon/label smoothing
- New FM baselines (MOMENT, Mantis) if added

### F3. Document Key Findings
- **R2_prod finding:** non-stationarity + FM local context advantage (legitimate contribution)
- **MASE correction:** training-data denominator per Hyndman & Koehler 2006
- **3W ceiling analysis:** Normal-Hydrate confusion, F1~0.964 may be label-noise ceiling
- **Literature context:** position results against 12+ papers using 3W dataset

### F4. Clean Up Orphaned Result Directories
- 7 result dirs not in statistical tests: chronos_finetuned, convtran, convtran_raw, inception_time, inception_time_raw, multiscale_deeponet, multiscale_rf
- Decision: archive or include in comparison

---

## Timeline Estimate

| Phase | When | Compute | Dependencies |
|-------|------|---------|-------------|
| A (wait) | This week | Running | None |
| B (analyze) | After HPC completes | 0 GPU-hrs | Phase A |
| C (ensemble) | +1-2 days | 0-2 GPU-hrs | Phase B (diversity gate) |
| D (improvements) | +1-2 weeks | 5-20 GPU-hrs | Phase B (analysis informs priority) |
| E (new FMs) | +1-2 weeks | 10-15 GPU-hrs | Docker rebuild |
| F (dissertation) | After all experiments | 0 | Phases B-E |

**Total additional compute:** 15-37 GPU-hours (fits in 1-2 HPC sessions on 2 nodes)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/offshore_dl/models/base.py` | Label smoothing parameter (already added) |
| `src/offshore_dl/training/supcon.py` | SupCon pre-training (already implemented) |
| `src/offshore_dl/models/moment_wrapper.py` | MOMENT FM wrapper (already implemented) |
| `src/offshore_dl/models/mantis_wrapper.py` | Mantis FM wrapper (already implemented) |
| `src/offshore_dl/training/experiment.py` | Retrain split fix (already applied) |
| `scripts/ensemble_3w.py` | 3W ensemble script |
| `scripts/sweep_utils.py` | Shared sweep utilities |
| `scripts/slurm_run_model.sh` | Parameterized HPC job |
| `scripts/deploy_mase_rerun.sh` | HPC deployment orchestrator |
| `.omc/plans/improve-results.md` | Full consensus plan (Planner/Architect/Critic approved) |
| `.omc/specs/deep-interview-improve-results.md` | Deep interview spec |
