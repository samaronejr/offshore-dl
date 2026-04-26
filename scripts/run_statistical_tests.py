"""Statistical significance tests across models.

Produces Friedman test (multi-model ranking) + Nemenyi post-hoc
and pairwise Wilcoxon signed-rank tests on inner CV fold-level metrics.

Reads nested CV results from results/ directory.
Outputs reports/statistical_tests_nested.json.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
REPORT_PATH = Path("reports") / "statistical_tests_nested.json"

# ── 3W classification models ──
CLS_MODELS = [
    "lstm",
    "deeponet",
    "patchtst",
    "convtran",
    "convtimenet",
    "mambasl",
    "fkmad",
    "random_forest",
    "wavelet_rf",
    "wavelet_deeponet",
    "physics_rf",
    "physics_deeponet",
]
CLS_METRICS = ["accuracy", "f1_macro", "auc_pr"]

# ── CDF anomaly models ──
CDF_MODELS = ["lstm", "deeponet", "patchtst", "chronos", "timesfm", "tirex"]
CDF_METRICS = ["error_mean", "error_p50"]

# ── Ganymede forecasting models (trained only for now) ──
FC_TRAINED = ["lstm", "deeponet", "patchtst", "tcn"]
FC_TREE = []
FC_FMS = ["chronos", "timesfm", "tirex"]
FC_METRICS = ["mae", "r2_prod"]
HORIZONS = [7, 14, 30, 90]

# ── SPE BERG forecasting models ──
SPE_BERG_TRAINED = ["lstm", "deeponet", "patchtst", "tcn"]
SPE_BERG_FMS = ["chronos", "timesfm", "tirex"]

# ── Volve forecasting models ──
VOLVE_TRAINED = ["lstm", "deeponet", "patchtst", "tcn"]
VOLVE_FMS = ["chronos", "timesfm", "tirex"]

# ── Inner Mongolia forecasting models ──
IM_TRAINED = ["lstm", "deeponet", "patchtst", "tcn"]
IM_FMS = ["chronos", "timesfm", "tirex"]


def _load_3w_folds(model: str) -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for a 3W model."""
    if model == "tirex":
        path = RESULTS_DIR / "tirex_3w_nested.json"
    else:
        path = RESULTS_DIR / model / "3w.json"

    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", [])
    if not folds:
        return None
    return {
        metric: [fr["metrics"][metric] for fr in folds]
        for metric in CLS_METRICS
        if all(metric in fr["metrics"] for fr in folds)
    }


def _load_ganymede_folds(model: str, horizon: int, mode: str = "multi_well") -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for a Ganymede model."""
    path = RESULTS_DIR / model / f"ganymede_h{horizon}_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", data.get("fold_results", []))
    if not folds:
        return None
    return {
        metric: [fr["metrics"].get(metric, fr["metrics"].get(metric, float("nan")))
                 for fr in folds]
        for metric in FC_METRICS
        if all(metric in fr.get("metrics", {}) for fr in folds)
    }


def _friedman_test(scores: dict[str, list[float]]) -> dict:
    """Run Friedman test on model scores (each value = list of fold scores).

    Requires ≥ 3 models and ≥ 3 folds.
    """
    models = sorted(scores.keys())
    n_models = len(models)
    if n_models < 3:
        return {"status": "insufficient_models", "n_models": n_models}

    arrays = [np.array(scores[m]) for m in models]
    n_folds = len(arrays[0])
    if n_folds < 3:
        return {"status": "insufficient_folds", "n_folds": n_folds}

    stat, p = stats.friedmanchisquare(*arrays)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "n_models": n_models,
        "n_folds": n_folds,
        "models": models,
    }


def _nemenyi_cd(n_models: int, n_folds: int, alpha: float = 0.05) -> float:
    """Compute Nemenyi critical difference.

    CD = q_α * sqrt(n_models * (n_models + 1) / (6 * n_folds))

    q_α values from Demšar (2006) Table 5 for α = 0.05.
    """
    # q_α for α=0.05, k models (studentized range / sqrt(2))
    q_table = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728,
        6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
        11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    q = q_table.get(n_models, q_table[max(q_table.keys())])  # fallback to largest known k
    return q * np.sqrt(n_models * (n_models + 1) / (6 * n_folds))


def _compute_ranks(scores: dict[str, list[float]], higher_is_better: bool = True) -> dict:
    """Compute average ranks across folds (with proper tie handling)."""
    models = sorted(scores.keys())
    arrays = np.array([scores[m] for m in models])  # (n_models, n_folds)
    n_models, n_folds = arrays.shape

    # Rank per fold (1 = best) using scipy.stats.rankdata for correct tie handling
    ranks = np.zeros_like(arrays, dtype=float)
    for j in range(n_folds):
        col = arrays[:, j]
        if higher_is_better:
            col = -col  # negate so rank 1 = highest value
        ranks[:, j] = stats.rankdata(col, method="average")

    avg_ranks = ranks.mean(axis=1)
    return {models[i]: float(avg_ranks[i]) for i in range(n_models)}


def _wilcoxon_pairwise(scores: dict[str, list[float]]) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests with Holm correction."""
    models = sorted(scores.keys())
    results = []
    raw_pvals = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = np.array(scores[models[i]])
            b = np.array(scores[models[j]])
            diff = a - b
            if np.all(diff == 0):
                results.append({
                    "model_a": models[i], "model_b": models[j],
                    "statistic": 0.0, "p_value": 1.0, "p_value_raw": 1.0,
                    "significant": False,
                    "note": "identical scores",
                })
                raw_pvals.append(1.0)
                continue
            try:
                stat, p = stats.wilcoxon(a, b)
                results.append({
                    "model_a": models[i], "model_b": models[j],
                    "statistic": float(stat), "p_value_raw": float(p),
                })
                raw_pvals.append(float(p))
            except ValueError as e:
                results.append({
                    "model_a": models[i], "model_b": models[j],
                    "error": str(e),
                })
                raw_pvals.append(1.0)

    # Apply Holm correction for multiple comparisons — only over valid Wilcoxon
    # p-values. Placeholder 1.0s appended for identical-score and ValueError
    # pairs must NOT inflate n (family size) in the step-down formula.
    if raw_pvals:
        from statsmodels.stats.multitest import multipletests
        valid_idx = [
            i for i, r in enumerate(results)
            if "error" not in r and "note" not in r
        ]
        if valid_idx:
            valid_p = [raw_pvals[i] for i in valid_idx]
            _, corrected, _, _ = multipletests(valid_p, method="holm")
            for k, i in enumerate(valid_idx):
                results[i]["p_value"] = float(corrected[k])
                results[i]["significant"] = bool(corrected[k] < 0.05)

    return results


def _run_3w_tests() -> dict:
    """Run statistical tests on 3W classification results."""
    logger.info("═══ 3W Classification Statistical Tests ═══")

    all_folds = {}
    for model in CLS_MODELS:
        folds = _load_3w_folds(model)
        if folds:
            all_folds[model] = folds
            logger.info("  %s: %d folds loaded", model, len(next(iter(folds.values()))))

    if len(all_folds) < 2:
        return {"status": "insufficient_models", "n_models": len(all_folds)}

    results = {"n_models": len(all_folds), "metrics": {}}

    for metric in CLS_METRICS:
        scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
        higher_is_better = True  # all classification metrics are higher=better

        friedman = _friedman_test(scores)
        ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
        wilcoxon = _wilcoxon_pairwise(scores)

        metric_result = {
            "models": sorted(scores.keys()),
            "friedman": friedman,
            "average_ranks": ranks,
            "wilcoxon_pairwise": wilcoxon,
        }

        if friedman.get("significant"):
            n_models = friedman["n_models"]
            n_folds = friedman["n_folds"]
            cd = _nemenyi_cd(n_models, n_folds)
            metric_result["nemenyi_cd"] = float(cd)
            # Check which pairs differ significantly
            sig_pairs = []
            models = sorted(ranks.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                    if rank_diff > cd:
                        sig_pairs.append({
                            "model_a": models[i], "model_b": models[j],
                            "rank_diff": float(rank_diff),
                        })
            metric_result["nemenyi_significant_pairs"] = sig_pairs

        results["metrics"][metric] = metric_result
        logger.info("  %s: Friedman p=%.4f sig=%s | ranks: %s",
                     metric, friedman.get("p_value", -1), friedman.get("significant"),
                     {k: f"{v:.2f}" for k, v in ranks.items()})

    return results


def _run_ganymede_tests() -> dict:
    """Run statistical tests on Ganymede forecasting results per horizon."""
    logger.info("═══ Ganymede Forecasting Statistical Tests ═══")

    results = {}
    all_models = FC_TRAINED + FC_TREE + FC_FMS

    for horizon in HORIZONS:
        all_folds = {}
        for model in all_models:
            folds = _load_ganymede_folds(model, horizon)
            if folds:
                all_folds[model] = folds

        if len(all_folds) < 2:
            results[f"h{horizon}"] = {"status": "insufficient_models", "n_models": len(all_folds)}
            logger.info("  h%d: only %d models with fold data", horizon, len(all_folds))
            continue

        logger.info("  h%d: %d models with fold data: %s", horizon, len(all_folds), list(all_folds.keys()))

        horizon_results = {"n_models": len(all_folds), "metrics": {}}

        for metric in FC_METRICS:
            scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
            if len(scores) < 2:
                continue

            higher_is_better = metric in ("r2", "r2_prod")  # MAE/RMSE lower=better

            friedman = _friedman_test(scores)
            ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
            wilcoxon = _wilcoxon_pairwise(scores)

            metric_result = {
                "models": sorted(scores.keys()),
                "friedman": friedman,
                "average_ranks": ranks,
                "wilcoxon_pairwise": wilcoxon,
            }

            if friedman.get("significant"):
                n_models = friedman["n_models"]
                n_folds = friedman["n_folds"]
                cd = _nemenyi_cd(n_models, n_folds)
                metric_result["nemenyi_cd"] = float(cd)
                sig_pairs = []
                models = sorted(ranks.keys())
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                        if rank_diff > cd:
                            sig_pairs.append({
                                "model_a": models[i], "model_b": models[j],
                                "rank_diff": float(rank_diff),
                            })
                metric_result["nemenyi_significant_pairs"] = sig_pairs

            horizon_results["metrics"][metric] = metric_result
            logger.info("    %s %s: Friedman p=%.4f sig=%s",
                         f"h{horizon}", metric,
                         friedman.get("p_value", -1), friedman.get("significant"))

        results[f"h{horizon}"] = horizon_results

    return results


def _load_spe_berg_folds(model: str, horizon: int, mode: str = "multi_well") -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for a SPE BERG model."""
    path = RESULTS_DIR / model / f"spe_berg_h{horizon}_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", data.get("fold_results", []))
    if not folds:
        return None
    return {
        metric: [fr["metrics"].get(metric, float("nan"))
                 for fr in folds]
        for metric in FC_METRICS
        if all(metric in fr.get("metrics", {}) for fr in folds)
    }


def _run_spe_berg_tests() -> dict:
    """Run statistical tests on SPE BERG forecasting results per horizon."""
    logger.info("═══ SPE BERG Forecasting Statistical Tests ═══")

    results = {}
    all_models = SPE_BERG_TRAINED + SPE_BERG_FMS

    for horizon in HORIZONS:
        all_folds = {}
        for model in all_models:
            folds = _load_spe_berg_folds(model, horizon)
            if folds:
                all_folds[model] = folds

        if len(all_folds) < 2:
            results[f"h{horizon}"] = {"status": "insufficient_models", "n_models": len(all_folds)}
            logger.info("  h%d: only %d models with fold data", horizon, len(all_folds))
            continue

        logger.info("  h%d: %d models with fold data: %s", horizon, len(all_folds), list(all_folds.keys()))

        horizon_results = {"n_models": len(all_folds), "metrics": {}}

        for metric in FC_METRICS:
            scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
            if len(scores) < 2:
                continue

            higher_is_better = metric in ("r2", "r2_prod")  # MAE/RMSE lower=better

            friedman = _friedman_test(scores)
            ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
            wilcoxon = _wilcoxon_pairwise(scores)

            metric_result = {
                "models": sorted(scores.keys()),
                "friedman": friedman,
                "average_ranks": ranks,
                "wilcoxon_pairwise": wilcoxon,
            }

            if friedman.get("significant"):
                n_models = friedman["n_models"]
                n_folds = friedman["n_folds"]
                cd = _nemenyi_cd(n_models, n_folds)
                metric_result["nemenyi_cd"] = float(cd)
                sig_pairs = []
                models = sorted(ranks.keys())
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                        if rank_diff > cd:
                            sig_pairs.append({
                                "model_a": models[i], "model_b": models[j],
                                "rank_diff": float(rank_diff),
                            })
                metric_result["nemenyi_significant_pairs"] = sig_pairs

            horizon_results["metrics"][metric] = metric_result
            logger.info("    %s %s: Friedman p=%.4f sig=%s",
                         f"h{horizon}", metric,
                         friedman.get("p_value", -1), friedman.get("significant"))

        results[f"h{horizon}"] = horizon_results

    return results


def _load_volve_folds(model: str, horizon: int, mode: str = "multi_well") -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for a Volve model."""
    path = RESULTS_DIR / model / f"volve_h{horizon}_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", data.get("fold_results", []))
    if not folds:
        return None
    return {
        metric: [fr["metrics"].get(metric, float("nan"))
                 for fr in folds]
        for metric in FC_METRICS
        if all(metric in fr.get("metrics", {}) for fr in folds)
    }


def _run_volve_tests() -> dict:
    """Run statistical tests on Volve forecasting results per horizon."""
    logger.info("═══ Volve Oil Forecasting Statistical Tests ═══")

    results = {}
    all_models = VOLVE_TRAINED + VOLVE_FMS

    for horizon in HORIZONS:
        all_folds = {}
        for model in all_models:
            folds = _load_volve_folds(model, horizon)
            if folds:
                all_folds[model] = folds

        if len(all_folds) < 2:
            results[f"h{horizon}"] = {"status": "insufficient_models", "n_models": len(all_folds)}
            logger.info("  h%d: only %d models with fold data", horizon, len(all_folds))
            continue

        logger.info("  h%d: %d models with fold data: %s", horizon, len(all_folds), list(all_folds.keys()))

        horizon_results = {"n_models": len(all_folds), "metrics": {}}

        for metric in FC_METRICS:
            scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
            if len(scores) < 2:
                continue

            higher_is_better = metric in ("r2", "r2_prod")  # MAE/RMSE lower=better

            friedman = _friedman_test(scores)
            ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
            wilcoxon = _wilcoxon_pairwise(scores)

            metric_result = {
                "models": sorted(scores.keys()),
                "friedman": friedman,
                "average_ranks": ranks,
                "wilcoxon_pairwise": wilcoxon,
            }

            if friedman.get("significant"):
                n_models = friedman["n_models"]
                n_folds = friedman["n_folds"]
                cd = _nemenyi_cd(n_models, n_folds)
                metric_result["nemenyi_cd"] = float(cd)
                sig_pairs = []
                models = sorted(ranks.keys())
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                        if rank_diff > cd:
                            sig_pairs.append({
                                "model_a": models[i], "model_b": models[j],
                                "rank_diff": float(rank_diff),
                            })
                metric_result["nemenyi_significant_pairs"] = sig_pairs

            horizon_results["metrics"][metric] = metric_result
            logger.info("    %s %s: Friedman p=%.4f sig=%s",
                         f"h{horizon}", metric,
                         friedman.get("p_value", -1), friedman.get("significant"))

        results[f"h{horizon}"] = horizon_results

    return results


def _load_inner_mongolia_folds(model: str, horizon: int, mode: str = "multi_well") -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for an Inner Mongolia model."""
    path = RESULTS_DIR / model / f"inner_mongolia_h{horizon}_{mode}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", data.get("fold_results", []))
    if not folds:
        return None
    return {
        metric: [fr["metrics"].get(metric, float("nan"))
                 for fr in folds]
        for metric in FC_METRICS
        if all(metric in fr.get("metrics", {}) for fr in folds)
    }


def _run_inner_mongolia_tests() -> dict:
    """Run statistical tests on Inner Mongolia forecasting results per horizon."""
    logger.info("═══ Inner Mongolia Gas Forecasting Statistical Tests ═══")

    results = {}
    all_models = IM_TRAINED + IM_FMS

    for horizon in HORIZONS:
        all_folds = {}
        for model in all_models:
            folds = _load_inner_mongolia_folds(model, horizon)
            if folds:
                all_folds[model] = folds

        if len(all_folds) < 2:
            results[f"h{horizon}"] = {"status": "insufficient_models", "n_models": len(all_folds)}
            logger.info("  h%d: only %d models with fold data", horizon, len(all_folds))
            continue

        logger.info("  h%d: %d models with fold data: %s", horizon, len(all_folds), list(all_folds.keys()))

        horizon_results = {"n_models": len(all_folds), "metrics": {}}

        for metric in FC_METRICS:
            scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
            if len(scores) < 2:
                continue

            higher_is_better = metric in ("r2", "r2_prod")  # MAE/RMSE lower=better

            friedman = _friedman_test(scores)
            ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
            wilcoxon = _wilcoxon_pairwise(scores)

            metric_result = {
                "models": sorted(scores.keys()),
                "friedman": friedman,
                "average_ranks": ranks,
                "wilcoxon_pairwise": wilcoxon,
            }

            if friedman.get("significant"):
                n_models = friedman["n_models"]
                n_folds = friedman["n_folds"]
                cd = _nemenyi_cd(n_models, n_folds)
                metric_result["nemenyi_cd"] = float(cd)
                sig_pairs = []
                models = sorted(ranks.keys())
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                        if rank_diff > cd:
                            sig_pairs.append({
                                "model_a": models[i], "model_b": models[j],
                                "rank_diff": float(rank_diff),
                            })
                metric_result["nemenyi_significant_pairs"] = sig_pairs

            horizon_results["metrics"][metric] = metric_result
            logger.info("    %s %s: Friedman p=%.4f sig=%s",
                         f"h{horizon}", metric,
                         friedman.get("p_value", -1), friedman.get("significant"))

        results[f"h{horizon}"] = horizon_results

    return results


def _load_cdf_folds(model: str) -> dict[str, list[float]] | None:
    """Load inner CV fold metrics for a CDF model."""
    path = RESULTS_DIR / model / "cdf.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    folds = data.get("cv_fold_results", [])
    if not folds:
        return None
    return {
        metric: [fr["metrics"][metric] for fr in folds]
        for metric in CDF_METRICS
        if all(metric in fr.get("metrics", {}) for fr in folds)
    }


def _run_cdf_tests() -> dict:
    """Run statistical tests on CDF anomaly detection results."""
    logger.info("═══ CDF Anomaly Detection Statistical Tests ═══")

    all_folds = {}
    for model in CDF_MODELS:
        folds = _load_cdf_folds(model)
        if folds:
            all_folds[model] = folds
            logger.info("  %s: %d folds loaded", model, len(next(iter(folds.values()))))

    if len(all_folds) < 2:
        return {"status": "insufficient_models", "n_models": len(all_folds)}

    results = {"n_models": len(all_folds), "metrics": {}}

    for metric in CDF_METRICS:
        scores = {m: all_folds[m][metric] for m in all_folds if metric in all_folds[m]}
        if len(scores) < 2:
            continue
        higher_is_better = False  # error metrics: lower is better

        friedman = _friedman_test(scores)
        ranks = _compute_ranks(scores, higher_is_better=higher_is_better)
        wilcoxon = _wilcoxon_pairwise(scores)

        metric_result = {
            "models": sorted(scores.keys()),
            "friedman": friedman,
            "average_ranks": ranks,
            "wilcoxon_pairwise": wilcoxon,
        }

        if friedman.get("significant"):
            n_models = friedman["n_models"]
            n_folds = friedman["n_folds"]
            cd = _nemenyi_cd(n_models, n_folds)
            metric_result["nemenyi_cd"] = float(cd)
            sig_pairs = []
            models = sorted(ranks.keys())
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    rank_diff = abs(ranks[models[i]] - ranks[models[j]])
                    if rank_diff > cd:
                        sig_pairs.append({
                            "model_a": models[i], "model_b": models[j],
                            "rank_diff": float(rank_diff),
                        })
            metric_result["nemenyi_significant_pairs"] = sig_pairs

        results["metrics"][metric] = metric_result
        logger.info("  %s: Friedman p=%.4f sig=%s | ranks: %s",
                     metric, friedman.get("p_value", -1), friedman.get("significant"),
                     {k: f"{v:.2f}" for k, v in ranks.items()})

    return results


def main():
    report = {
        "description": "Statistical significance tests on inner CV fold-level results (nested evaluation)",
        "tests": ["Friedman chi-square (multi-model)", "Nemenyi post-hoc (if Friedman significant)", "Wilcoxon signed-rank (pairwise)"],
        "alpha": 0.05,
        "3w": _run_3w_tests(),
        "ganymede": _run_ganymede_tests(),
        "spe_berg": _run_spe_berg_tests(),
        "volve": _run_volve_tests(),
        "inner_mongolia": _run_inner_mongolia_tests(),
        "cdf": _run_cdf_tests(),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Results saved: %s", REPORT_PATH)


if __name__ == "__main__":
    main()
