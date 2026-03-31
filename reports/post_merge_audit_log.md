# Post-Merge Audit & Re-Run Log

**Date:** 2026-03-22/23  
**Branch:** `main`  
**Previous HEAD:** `28aaab4` (M002 squash merge)  
**Current HEAD:** `d0195f6` (audit fixes)

---

## 1. Context

After completing the M002 milestone (6 models × 3 datasets benchmark), an
audit was performed identifying 30 issues: 4 critical, 14 high, 12 medium. All fixes
were applied and verified. This document tracks the full post-merge workflow: audit
integration, re-runs with corrected code, and final paper update.

## 2. Audit Fixes Applied (commit `d0195f6`)

### Critical Bugs Fixed
| ID | Issue | Impact |
|----|-------|--------|
| C1 | `compare.py` read `fold_results` but JSON uses `cv_fold_results` | All auto-generated LaTeX tables and statistical test JSONs were stale/wrong |
| C2 | `GanymedeDataset` computed target column index before `sorted()` alignment | Could forecast wrong sensor column after multi-well alignment |
| C3 | No best-model restoration after early stopping in `Trainer.fit()` | Models evaluated with overfit last-epoch weights instead of best-epoch |
| C4 | `statistical_tests.json` unreproducible | Consequence of C1 |

### High-Priority Fixes
| ID | Issue | Fix |
|----|-------|-----|
| H1 | LeakageGuard never called | Added set-intersection check in `_run_fold()` |
| H2 | 3W used CSV-based CV (v1.x incompatible) | Replaced with `StratifiedGroupKFoldSKLearn` |
| H3 | Friedman test with n=3 < k=6 | Added sample-size warning |
| H4 | Wilcoxon min p > 0.05 with n=3 | Added structural warning |
| H6 | MASE denominator from test, not train | Added `y_train` param (Hyndman & Koehler) |
| H7 | AUC-PR from hard labels, not probabilities | Added `prediction_scores` param |
| H12 | PatchTST `e_layers` silently ignored | Changed to `n_layers` |
| H13 | PatchTST hardcoded channel 0 | Added `target_channel` kwarg |

### Medium Fixes
| ID | Issue | Fix |
|----|-------|-----|
| M4 | No multiple comparisons correction | Holm correction on pairwise Wilcoxon |
| M6 | No effect sizes | Kendall's W for Friedman |
| M9 | TiRex CDF was constant-mean baseline | Uses actual TiRex model now |

### Code Quality
- All new parameters default to `None` — backward compatible
- 19 files changed, 608 insertions, 59 deletions
- Full audit report: `reports/audit_fix_report.md`
- Decisions D48–D61 recorded in `.gsd/DECISIONS.md`

## 3. Infrastructure Fixes

### Data Symlink
The M002 squash merge created a circular symlink (`data → data`). Fixed by:
1. Removing the circular symlink
2. Creating `data/` directory with symlinks to actual data locations:
   - `data/raw/3w/` → `/home/samarone/Documents/msc_dissertacao/data/raw/3w/`
   - `data/raw/CogniteDataFusion/` → CDF sensor CSV
   - `data/raw/NSTA_datasets/cleaned/` → Ganymede CSV
3. Removed `data` from git tracking (was committed as symlink)
4. Added `data/` to `.gitignore`

### Docker Rebuild
Full `--no-cache` rebuild of `offshore-dl:train` with all audit fixes.

### Worktree Cleanup
Worktree at `.gsd/worktrees/M002/` has root-owned files from Docker.
Needs `sudo rm -rf .gsd/worktrees/M002` to fully clean.

## 4. Test Results After Fixes

```
266 passed, 10 skipped, 1218 warnings in 26.14s
```

- 10 skipped: 8 FM dependency tests + 2 Ganymede (need preprocessing run)
- All code-level tests pass including new audit-related assertions

## 5. Production Re-Runs

### Local Docker (killed)
Initial attempt ran locally but killed in favor of HPC parallelism.

### LPS HPC Cluster (in progress)
Deploying to LPS/UFRJ Slurm cluster for parallel execution on 3 separate GPU nodes:

| Job | Script | Node | GPU | Est. Time |
|-----|--------|------|-----|-----------|
| 3W classification | `slurm_3w.sh` | any GPU | RTX 2080S/3090 | ~3 hrs |
| CDF anomaly | `slurm_cdf.sh` | any GPU | RTX 2080S/3090 | ~1 hr |
| Ganymede forecast | `slurm_ganymede.sh` | any GPU | RTX 2080S/3090 | ~6 hrs |

All 3 run simultaneously on different nodes instead of 8+ hrs sequential on local GPU.

Deploy script: `scripts/deploy_lps.sh`
Monitor: `ssh LPS_loginServer 'squeue -u $USER'`
Fetch results: `rsync -avz LPS_loginServer:~/offshore-dl/results/ results/`

### 3W NaN Investigation (RESOLVED)
All 3W trained models produced NaN from epoch 1 — both locally and on HPC.
Confirmed NOT caused by audit fixes (same NaN with pre-audit code `28aaab4`).

**Root cause**: The raw 3W parquets contain **frozen sensor values** — pressure
sensors reading constant `1e7 Pa` for thousands of consecutive timesteps. These
are not missing data sentinels but physically frozen readings that the `preprocess_3w`
pipeline was designed to detect and replace with NaN → forward fill.

When running directly on raw data (the fallback path when `data/processed/3w/` doesn't
exist), the frozen 1e7 values produce extreme statistical features:
- Energy (sum x²) = `720 × (1e7)² = 7.2e16`
- These dominate normalization, producing numerically unstable gradients → NaN

**Fix**: Run `preprocess_3w()` to create `data/processed/3w/`. The frozen-value
detection replaces constant readings with NaN, producing clean features.
Training confirmed working: `train_loss=0.016, val_loss=1e-6` (no NaN).

Also: changed Docker data mount from `:ro` to read-write so preprocessing
can create `data/processed/` inside the container.

## 6. Optuna HPO Status

### Completed
- **LSTM 3W**: 15 trials, best hidden=128/layers=1/dropout=0.31/lr=7.9e-5
  - Test: acc=96.62%, F1m=0.960 (baseline: 96.70%, 0.963)
  - Finding: smaller model (4× fewer params) matches baseline

### DeepONet 3W
- 15 trials completed, best F1=0.954 in CV
- Final eval collapsed (acc=15.8%) due to epoch mismatch (100 vs 50)
- **Fixed**: `max_epochs` now 50 in final eval (matches trials)
- Needs re-run after production baselines complete

### PatchTST 3W
- Killed mid-run for merge. Needs restart after baselines.

### Ganymede HPO
- Script ready but trained models far behind FMs — low priority.

## 7. Paper Status

### Current State (commit `2e641df`)
- 6 pages IEEE format
- All 3 datasets covered: 3W, Ganymede, CDF
- CDF section added with table, discussion, significance
- Abstract and contributions updated for 3-dataset scope
- All table numbers verified against JSON results

### Post-Re-Run Updates Needed
- Update any tables where numbers change from audit fixes
- Verify significance test results after re-generation
- Add HPO results if materially different from baselines

## 8. Remaining Work

### Immediate (after re-runs complete)
- [ ] Compare new vs old results, document changes
- [ ] Update paper tables if numbers changed
- [ ] Regenerate statistical tests report
- [ ] Commit new results

### Short-term
- [ ] Re-run DeepONet HPO final eval (epoch fix)
- [ ] Run PatchTST HPO (15 trials)
- [ ] `sudo rm -rf .gsd/worktrees/M002` (cleanup)

### Dissertation
- [ ] Discuss Friedman power limitations (H3/H4)
- [ ] Discuss FM anomaly semantics (H9/H11)
- [ ] Discuss zero-shot vs trained fairness (H10)
- [ ] Discuss gap=0 adjacent-sample overlap (M1)
- [ ] Discuss normalization scale differences (M2/M3)

---

## Appendix: Key Commits

| Hash | Description |
|------|-------------|
| `28aaab4` | Squash merge M002 → main (70 commits, 341 files) |
| `d0195f6` | Audit fixes: 30 findings, 19 files, 608 insertions |
| `2e641df` | Paper update: 3-dataset scope, CDF throughout |
| `9fb64e9` | Project review: README, stale files, baseline archive |
| `f9b0f6f` | CDF section + updated stats |
| `157595f` | LSTM HPO complete |
