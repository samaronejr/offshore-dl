"""Populate SPE BERG LaTeX tables in reports/all_results.tex.

Reads result JSONs from results/{model}/spe_berg_h{H}_multi_well.json and
the Friedman stats from reports/statistical_tests_nested.json, then replaces
the placeholder rows in the three SPE BERG tables:
  - tab:spe_berg_main    (held-out test r2_prod + MAE per model x horizon)
  - tab:spe_berg_cv      (CV mean+-std per model x horizon)
  - tab:spe_berg_friedman (Friedman chi2 + p + rank order per horizon)

Also fills in placeholder cdots for SPE BERG h=7 columns in tab:summary_all.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
TEX_PATH = Path("reports") / "all_results.tex"
STAT_PATH = Path("reports") / "statistical_tests_nested.json"
HORIZONS = [7, 14, 30, 90]

MODEL_ORDER = ["timesfm", "tirex", "chronos", "patchtst", "lstm", "tcn", "deeponet"]
DISPLAY_NAMES = {
    "timesfm": "TimesFM",
    "tirex": "TiRex",
    "chronos": "Chronos",
    "patchtst": "PatchTST",
    "lstm": "LSTM",
    "tcn": "TCN",
    "deeponet": "DeepONet",
}
FM_MODELS = {"timesfm", "tirex", "chronos"}


# ── Number formatting ─────────────────────────────────────────────────────────

def fmt_r2(v: float | None) -> str:
    """Format r2_prod: no leading zero, negative with $-$ prefix."""
    if v is None:
        return r"$\cdots$"
    if abs(v) < 1e-9:
        return ".000"
    if v < 0:
        mag = abs(v)
        if mag >= 10:
            return r"$-$" + f"{mag:.1f}"
        return r"$-$" + f"{mag:.3f}"
    # Positive: strip leading zero if < 1
    s = f"{v:.3f}"
    if s.startswith("0."):
        s = s[1:]  # ".NNN"
    return s


def fmt_mae(v: float | None) -> str:
    """Format MAE: 3dp, strip leading zero if < 1, negative with $-$."""
    if v is None:
        return r"$\cdots$"
    if v < 0:
        return r"$-$" + f"{abs(v):.3f}"
    s = f"{v:.3f}"
    if s.startswith("0."):
        s = s[1:]
    return s


def fmt_cv(mean: float | None, std: float | None, is_r2: bool) -> str:
    """Format CV aggregate as mean±std.

    For r2_prod, values use $-$ prefix for negatives (already formatted by fmt_r2).
    For MAE, values are always positive in practice, but fmt_mae handles negatives.
    The output is wrapped in a single math environment; we must not double-wrap.
    """
    if mean is None or std is None:
        return r"$\cdots$"
    fmt = fmt_r2 if is_r2 else fmt_mae
    m_s = fmt(mean)
    # std: always 2dp, strip leading zero
    s_s = f"{std:.2f}"
    if s_s.startswith("0."):
        s_s = s_s[1:]
    # If mean string already contains $ (e.g. "$-$0.118"), we need to embed without extra $
    if "$" in m_s:
        # e.g. "$-$0.118" → build: $-$0.118 \pm .04 but inside one math env
        # Strip outer $ if present to build a clean math env
        # fmt_r2 returns "$-$X.XXX" for negatives — we want "-0.118 \pm .04" in math
        # Actually "$-$" is a LaTeX trick for minus sign in math mode.
        # Just concatenate directly without extra $ wrapping
        return m_s + r" $\pm$ " + s_s
    else:
        # positive values: m_s is already a plain string like ".327" or "1.234"
        return f"${m_s} \\pm {s_s}$"


def fmt_friedman_p(p: float) -> str:
    """Format p-value: 3 sig figs, or <0.001 for very small."""
    if p < 0.001:
        return r"$<$0.001"
    return f"{p:.3f}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_result(model: str, horizon: int) -> dict | None:
    """Load result JSON for model×horizon. Returns None if missing."""
    path = RESULTS_DIR / model / f"spe_berg_h{horizon}_multi_well.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(data: dict | None) -> dict:
    """Extract test_metrics and cv_aggregate from result, returning dict with None for missing."""
    if data is None or data.get("status") == "unavailable":
        return {
            "r2_prod": None, "mae": None,
            "r2_prod_mean": None, "r2_prod_std": None,
            "mae_mean": None, "mae_std": None,
            "available": False,
        }
    tm = data.get("test_metrics", {})
    cv = data.get("cv_aggregate", {})
    return {
        "r2_prod": tm.get("r2_prod"),
        "mae": tm.get("mae"),
        "r2_prod_mean": cv.get("r2_prod_mean"),
        "r2_prod_std": cv.get("r2_prod_std"),
        "mae_mean": cv.get("mae_mean"),
        "mae_std": cv.get("mae_std"),
        "available": True,
    }


# ── Best-value detection ──────────────────────────────────────────────────────

def find_best(values: list[float | None], higher_is_better: bool) -> int | None:
    """Return index of best value (ignoring None). Returns None if all missing."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if higher_is_better:
        return max(valid, key=lambda x: x[1])[0]
    else:
        return min(valid, key=lambda x: x[1])[0]


def wrap_best(s: str, is_best: bool) -> str:
    """Wrap string with \\best{} if it's the best value."""
    if is_best and s not in (r"N/A", r"$\cdots$"):
        return r"\best{" + s + "}"
    return s


# ── Table builders ────────────────────────────────────────────────────────────

def build_main_table(all_metrics: dict[str, dict[str, dict]]) -> str:
    """Build replacement rows for tab:spe_berg_main.

    all_metrics[model][f'h{H}'] = {r2_prod, mae, available, ...}
    """
    lines = []

    # Collect values per metric × horizon for best detection
    r2_vals = {h: [all_metrics[m][f"h{h}"]["r2_prod"] for m in MODEL_ORDER] for h in HORIZONS}
    mae_vals = {h: [all_metrics[m][f"h{h}"]["mae"] for m in MODEL_ORDER] for h in HORIZONS}
    r2_best = {h: find_best(r2_vals[h], higher_is_better=True) for h in HORIZONS}
    mae_best = {h: find_best(mae_vals[h], higher_is_better=False) for h in HORIZONS}

    for i, model in enumerate(MODEL_ORDER):
        fm_suffix = r"\fm" if model in FM_MODELS else ""
        display = DISPLAY_NAMES[model] + fm_suffix

        r2_cells = []
        mae_cells = []
        for h in HORIZONS:
            mdata = all_metrics[model][f"h{h}"]
            if not mdata["available"]:
                r2_cells.append("N/A")
                mae_cells.append("N/A")
            else:
                r2_s = fmt_r2(mdata["r2_prod"])
                mae_s = fmt_mae(mdata["mae"])
                best_idx_r2 = r2_best[h]
                best_idx_mae = mae_best[h]
                r2_cells.append(wrap_best(r2_s, best_idx_r2 == i))
                mae_cells.append(wrap_best(mae_s, best_idx_mae == i))

        r2_row = " & ".join(r2_cells)
        mae_row = " & ".join(mae_cells)

        lines.append(r"\multirow{2}{*}{" + display + "}")
        lines.append(r" & $\rp$ & " + r2_row + r" \\")
        lines.append(r" & MAE   & " + mae_row + r" \\")
        lines.append(r"\midrule" if i < len(MODEL_ORDER) - 1 else r"\bottomrule")

    return "\n".join(lines)


def build_cv_table(all_metrics: dict[str, dict[str, dict]]) -> str:
    """Build replacement rows for tab:spe_berg_cv."""
    lines = []

    # r2_prod block
    r2_mean_vals = {h: [all_metrics[m][f"h{h}"]["r2_prod_mean"] for m in MODEL_ORDER] for h in HORIZONS}
    r2_mean_best = {h: find_best(r2_mean_vals[h], higher_is_better=True) for h in HORIZONS}

    for i, model in enumerate(MODEL_ORDER):
        fm_suffix = r"\fm" if model in FM_MODELS else ""
        display = DISPLAY_NAMES[model] + fm_suffix
        cells = []
        for h in HORIZONS:
            mdata = all_metrics[model][f"h{h}"]
            if not mdata["available"]:
                # Use N/A for unavailable (e.g. TimesFM/TiRex with status=unavailable),
                # keep $\cdots$ only if file simply doesn't exist yet (None from load).
                data = load_result(model, h)
                if data is not None and data.get("status") == "unavailable":
                    cells.append("N/A")
                else:
                    cells.append(r"$\cdots$")
            else:
                s = fmt_cv(mdata["r2_prod_mean"], mdata["r2_prod_std"], is_r2=True)
                if r2_mean_best[h] == i and mdata["r2_prod_mean"] is not None:
                    s = r"\best{" + s + "}"
                cells.append(s)
        lines.append(f"{display:<12}  & $\\rp$ & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    # MAE block
    mae_mean_vals = {h: [all_metrics[m][f"h{h}"]["mae_mean"] for m in MODEL_ORDER] for h in HORIZONS}
    mae_mean_best = {h: find_best(mae_mean_vals[h], higher_is_better=False) for h in HORIZONS}

    for i, model in enumerate(MODEL_ORDER):
        fm_suffix = r"\fm" if model in FM_MODELS else ""
        display = DISPLAY_NAMES[model] + fm_suffix
        cells = []
        for h in HORIZONS:
            mdata = all_metrics[model][f"h{h}"]
            if not mdata["available"]:
                data = load_result(model, h)
                if data is not None and data.get("status") == "unavailable":
                    cells.append("N/A")
                else:
                    cells.append(r"$\cdots$")
            else:
                s = fmt_cv(mdata["mae_mean"], mdata["mae_std"], is_r2=False)
                if mae_mean_best[h] == i and mdata["mae_mean"] is not None:
                    s = r"\best{" + s + "}"
                cells.append(s)
        lines.append(f"{display:<12}  & MAE & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    return "\n".join(lines)


def build_friedman_table(stat_data: dict) -> str:
    """Build replacement rows for tab:spe_berg_friedman."""
    lines = []
    spe = stat_data.get("spe_berg", {})

    horizon_labels = {7: r"$h{=}7$", 14: r"$h{=}14$", 30: r"$h{=}30$", 90: r"$h{=}90$"}

    for h in HORIZONS:
        hdata = spe.get(f"h{h}", {})
        metrics = hdata.get("metrics", {})

        mae_data = metrics.get("mae", {})
        r2_data = metrics.get("r2_prod", {})

        friedman = mae_data.get("friedman", {})
        stat = friedman.get("statistic")
        p_val = friedman.get("p_value")

        if stat is None or friedman.get("status") in ("insufficient_models", "insufficient_folds"):
            stat_s = r"$\cdots$"
            p_s = r"$\cdots$"
        else:
            stat_s = f"{stat:.2f}"
            p_s = fmt_friedman_p(p_val)

        # Rank orders
        r2_ranks = r2_data.get("average_ranks", {})
        mae_ranks = mae_data.get("average_ranks", {})

        def rank_order(ranks: dict, higher_is_better: bool) -> str:
            if not ranks:
                return r"$\cdots$"
            sorted_models = sorted(ranks.keys(), key=lambda m: ranks[m])
            names = [DISPLAY_NAMES.get(m, m) for m in sorted_models]
            return ", ".join(names)

        r2_order = rank_order(r2_ranks, higher_is_better=True)
        mae_order = rank_order(mae_ranks, higher_is_better=False)

        label = horizon_labels[h]
        lines.append(f"{label:<10} & {stat_s} & {p_s} & {r2_order} & {mae_order} \\\\")

    lines.append(r"\bottomrule")
    return "\n".join(lines)


# ── LaTeX replacement helpers ─────────────────────────────────────────────────

def replace_table_body(tex: str, label: str, new_body: str) -> str:
    """Replace data rows in a LaTeX table identified by \\label{label}.

    Keeps everything up to and including \\midrule after \\label, replaces
    all rows until \\bottomrule, then appends \\end{tabular}\\end{table}.
    """
    # Find the label
    label_tok = r"\label{" + label + "}"
    idx = tex.find(label_tok)
    if idx == -1:
        raise ValueError(f"Label not found: {label}")

    # Find \midrule after the label (the first data separator)
    midrule_idx = tex.find(r"\midrule", idx)
    if midrule_idx == -1:
        raise ValueError(f"\\midrule not found after {label}")

    # The "before" part ends just after the \midrule + newline
    before_end = midrule_idx + len(r"\midrule") + 1  # +1 for \n

    # Find \end{tabular} after the midrule
    end_tabular_idx = tex.find(r"\end{tabular}", midrule_idx)
    if end_tabular_idx == -1:
        raise ValueError(f"\\end{{tabular}} not found after {label}")

    before = tex[:before_end]
    after = tex[end_tabular_idx:]
    return before + new_body + "\n" + after


def replace_summary_spe_berg_h7(tex: str, all_metrics: dict[str, dict[str, dict]]) -> str:
    r"""Replace placeholder cdots in tab:summary_all SPE BERG h=7 columns.

    The summary table has columns: ... | rp↑ | MAE↓ | ... for SPE BERG h=7.
    Each model row that has real h7 data gets its $\cdots$ replaced.
    Models in summary order from the existing tex.
    """
    # Map display name → r2_prod, mae at h7
    h7_data = {}
    for model in MODEL_ORDER:
        mdata = all_metrics[model]["h7"]
        if mdata["available"] and mdata["r2_prod"] is not None:
            h7_data[DISPLAY_NAMES[model]] = {
                "r2_prod": mdata["r2_prod"],
                "mae": mdata["mae"],
            }

    # Find best among available models for SPE BERG h7
    available_r2 = [v["r2_prod"] for v in h7_data.values() if v["r2_prod"] is not None]
    available_mae = [v["mae"] for v in h7_data.values() if v["mae"] is not None]
    best_r2 = max(available_r2) if available_r2 else None
    best_mae = min(available_mae) if available_mae else None

    # For each model line that has $\cdots$ in the SPE BERG columns, replace
    lines = tex.split("\n")
    new_lines = []
    in_summary = False
    summary_label_seen = False

    for line in lines:
        if r"\label{tab:summary_all}" in line:
            summary_label_seen = True
        if summary_label_seen and r"\end{table}" in line:
            in_summary = False

        if summary_label_seen and r"\midrule" in line and not in_summary:
            in_summary = True  # first midrule = data start

        if in_summary and r"$\cdots$" in line:
            # Determine which display name this row belongs to
            matched_name = None
            for dname in h7_data:
                # Match start of line (with optional backslash-based prefixes)
                if dname in line:
                    matched_name = dname
                    break

            if matched_name and matched_name in h7_data:
                mdata = h7_data[matched_name]
                r2_val = mdata["r2_prod"]
                mae_val = mdata["mae"]

                r2_s = fmt_r2(r2_val)
                mae_s = fmt_mae(mae_val)

                # Wrap best
                if best_r2 is not None and abs(r2_val - best_r2) < 1e-9:
                    r2_s = r"\best{" + r2_s + "}"
                if best_mae is not None and abs(mae_val - best_mae) < 1e-9:
                    mae_s = r"\best{" + mae_s + "}"

                # Replace: the line has two consecutive $\cdots$ for SPE BERG h7 columns.
                # Pattern: ...& $\cdots$ & $\cdots$ & ...
                # We replace the first pair of $\cdots$ that correspond to SPE BERG h7.
                # The SPE BERG columns are the 6th and 7th ampersand-delimited fields.
                parts = line.split("&")
                if len(parts) >= 7:
                    # columns: model(0) | 3W acc(1) | 3W f1(2) | gany r2(3) | gany mae(4) | spe r2(5) | spe mae(6) | cdf(7)
                    parts[5] = f" {r2_s} "
                    parts[6] = f" {mae_s} "
                    line = "&".join(parts)

        new_lines.append(line)

    return "\n".join(new_lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Load all metrics
    all_metrics: dict[str, dict[str, dict]] = {}
    for model in MODEL_ORDER:
        all_metrics[model] = {}
        for h in HORIZONS:
            data = load_result(model, h)
            all_metrics[model][f"h{h}"] = extract_metrics(data)
            avail = all_metrics[model][f"h{h}"]["available"]
            logger.info("  %s h%d: available=%s  r2_prod=%s  mae=%s",
                        model, h, avail,
                        all_metrics[model][f"h{h}"]["r2_prod"],
                        all_metrics[model][f"h{h}"]["mae"])

    # Load statistical test results
    if not STAT_PATH.exists():
        logger.error("Statistical test results not found: %s", STAT_PATH)
        sys.exit(1)
    with open(STAT_PATH) as f:
        stat_data = json.load(f)

    # Build table bodies
    main_body = build_main_table(all_metrics)
    cv_body = build_cv_table(all_metrics)
    friedman_body = build_friedman_table(stat_data)

    logger.info("Built main table body (%d lines)", main_body.count("\n") + 1)
    logger.info("Built CV table body (%d lines)", cv_body.count("\n") + 1)
    logger.info("Built Friedman table body (%d lines)", friedman_body.count("\n") + 1)

    # Read existing tex
    tex = TEX_PATH.read_text()

    # Replace table bodies
    tex = replace_table_body(tex, "tab:spe_berg_main", main_body)
    logger.info("Replaced tab:spe_berg_main")

    tex = replace_table_body(tex, "tab:spe_berg_cv", cv_body)
    logger.info("Replaced tab:spe_berg_cv")

    tex = replace_table_body(tex, "tab:spe_berg_friedman", friedman_body)
    logger.info("Replaced tab:spe_berg_friedman")

    # Replace SPE BERG h7 columns in summary table
    tex = replace_summary_spe_berg_h7(tex, all_metrics)
    logger.info("Updated tab:summary_all SPE BERG h=7 columns")

    # Write modified tex
    TEX_PATH.write_text(tex)
    logger.info("Written: %s", TEX_PATH)

    # Verify no $\cdots$ in SPE BERG section
    try:
        spe_start = tex.index("SPE BERG Gas Production")
        spe_end = tex.index("CDF Anomaly Detection")
        spe_section = tex[spe_start:spe_end]
        cdots_count = spe_section.count("cdots")
        if cdots_count > 0:
            logger.warning("WARNING: %d \\cdots remain in SPE BERG section", cdots_count)
            # Find them
            for i, line in enumerate(spe_section.split("\n"), 1):
                if "cdots" in line:
                    logger.warning("  Line %d: %s", i, line.strip())
        else:
            logger.info("PASS: no \\cdots remain in SPE BERG section")
    except ValueError as e:
        logger.warning("Could not verify section bounds: %s", e)


if __name__ == "__main__":
    main()
