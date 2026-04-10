# GPU-Waiting Rerun Checklist

This file is the operational checklist to use while local/HPC GPUs are occupied.

## 1) CPU-safe work to do now

### Claim hygiene
- Treat current forecasting benchmark numbers as **stale until rerun**.
- Treat 3W deep-model `auc_pr` / `edr` numbers as **stale until rerun**.
- Do **not** cite old multi-well forecasting tables as final.

### Text to update now
- Thesis / dissertation chapters that describe:
  - multi-well forecasting split protocol
  - shutdown filtering policy
  - AUC-PR and EDR computation for 3W classification
- Presentation slides / README snippets / notes that quote old benchmark numbers.

### CPU-safe smoke commands
These are not full benchmarks; they only validate the corrected pipeline shape.

```bash
python scripts/run_production_ganymede.py --device cpu --max-epochs 1 --models lstm --no-mlflow
python scripts/run_production_spe_berg.py --device cpu --max-epochs 1 --models lstm --multi-well-only --no-mlflow
python scripts/run_production_volve.py --device cpu --max-epochs 1 --models lstm --multi-well-only --no-mlflow
python scripts/run_production_inner_mongolia.py --device cpu --max-epochs 1 --models lstm --multi-well-only --no-mlflow
```

Expected outcome:
- grouped temporal holdout runs end-to-end
- grouped inner CV runs end-to-end
- no shutdown-prefilter assumption breaks the pipeline

## 2) GPU rerun priority order

Run in this order once GPUs are free:

1. `ganymede`
2. `spe_berg`
3. `volve`
4. `inner_mongolia`
5. affected forecasting HPO jobs
6. any 3W deep-model reruns needed to refresh `auc_pr` / `edr`
7. statistical tests and regenerated tables / manuscript artifacts

Rationale:
- `ganymede` and `spe_berg` are directly implicated by the methodology bug review.
- `volve` and `inner_mongolia` use the same flawed flat-index multi-well temporal logic and should be refreshed for consistency.

## 3) Exact rerun commands

### Forecasting production sweeps

```bash
python scripts/run_production_ganymede.py --device cuda
python scripts/run_production_spe_berg.py --device cuda
python scripts/run_production_volve.py --device cuda
python scripts/run_production_inner_mongolia.py --device cuda
```

If you need to resume partially completed runs:

```bash
python scripts/run_production_spe_berg.py --device cuda --skip-existing
python scripts/run_production_volve.py --device cuda --skip-existing
python scripts/run_production_inner_mongolia.py --device cuda --skip-existing
```

### Focused multi-well-first reruns

Use this if you want to refresh the thesis-critical aggregate benchmarks before per-well outputs:

```bash
python scripts/run_production_spe_berg.py --device cuda --multi-well-only
python scripts/run_production_volve.py --device cuda --multi-well-only
python scripts/run_production_inner_mongolia.py --device cuda --multi-well-only
```

### Forecasting HPO reruns

Run only if you rely on HPO-derived best params/results that were computed with the old split logic.

Examples:

```bash
python scripts/run_optuna_hpo.py --dataset ganymede --models lstm deeponet patchtst tcn --horizon h7 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset ganymede --models lstm deeponet patchtst tcn --horizon h14 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset ganymede --models lstm deeponet patchtst tcn --horizon h30 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset ganymede --models lstm deeponet patchtst tcn --horizon h90 --n-trials 30 --device cuda

python scripts/run_optuna_hpo.py --dataset spe_berg --models lstm deeponet patchtst tcn --horizon h7 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset spe_berg --models lstm deeponet patchtst tcn --horizon h14 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset spe_berg --models lstm deeponet patchtst tcn --horizon h30 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset spe_berg --models lstm deeponet patchtst tcn --horizon h90 --n-trials 30 --device cuda

python scripts/run_optuna_hpo.py --dataset volve --models lstm deeponet patchtst tcn --horizon h7 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset volve --models lstm deeponet patchtst tcn --horizon h14 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset volve --models lstm deeponet patchtst tcn --horizon h30 --n-trials 30 --device cuda
python scripts/run_optuna_hpo.py --dataset volve --models lstm deeponet patchtst tcn --horizon h90 --n-trials 30 --device cuda
```

## 4) Post-rerun regeneration steps

After the benchmark reruns finish:

1. Regenerate downstream summaries / tables / manuscript-facing artifacts.
2. Re-run statistical tests:

```bash
python scripts/run_statistical_tests.py
```

3. Rebuild any LaTeX/PDF outputs that depend on refreshed result JSONs.
4. Re-check README headline numbers and thesis narrative claims.

## 5) Result audit checklist

After reruns, verify:
- multi-well held-out test is no longer concentrated in only the final wells
- benchmark counts changed in ways consistent with removing shutdown prefiltering
- 3W deep-model `auc_pr` differs from the old hard-label-derived values
- 3W deep-model `edr` is no longer trivially locked by the fallback proxy behavior
- retrain metadata shows disjoint retrain-train / retrain-val counts

## 6) Suggested evidence to archive

For each refreshed benchmark family, save:
- command used
- git commit SHA
- result JSON paths
- whether HPO was rerun or reused
- any changed headline numbers compared to the previous manuscript draft

## 7) Stop conditions

Do **not** update final thesis tables/claims until all of the following are true:
- corrected reruns finished
- downstream statistical tests reran
- affected paper/report tables regenerated
- README / manuscript text updated to match the corrected protocol
