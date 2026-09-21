#!/usr/bin/env python3
"""Redraw the available manuscript panels from the public aggregate CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANALYTES = ["GA", "OPF", "CAT", "ALB", "PF", "GPF", "BPF"]
TEAL, ORANGE, BLUE = "#287C8E", "#CC7544", "#5477A5"


def load(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing public figure data: {path.name}")
    return pd.read_csv(path, encoding="utf-8-sig")


def forest(ax, frame, value, low, high, *, y=None, color=TEAL, scale=1.0, label=None):
    center = frame[value].to_numpy(float) * scale
    lower, upper = frame[low].to_numpy(float) * scale, frame[high].to_numpy(float) * scale
    if y is None:
        y = np.arange(len(frame))
    # Draw intervals directly so percentile CIs need not contain their point estimates.
    ax.hlines(y, lower, upper, color=color, linewidth=1.5)
    ax.scatter(center, y, color=color, s=25, label=label, zorder=3)
    ax.axvline(0, color="#B9C1C5", linewidth=0.8)


def save(fig, output_dir: Path, stem: str):
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure2(data_dir, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    ax = axes[0, 0]
    frame = load(data_dir, "Figure_2A_HPLC_range_qualification.csv").sort_values("plot_order")
    x = np.arange(len(frame))
    for offset, (column, label, color) in enumerate([
        ("n_observed", "Observed", BLUE), ("n_full_range", "All-level range", ORANGE),
        ("n_strict_retained_range", "Retained range", TEAL),
    ]):
        ax.bar(x + (offset - 1) * .24, frame[column], width=.23, label=label, color=color)
    ax.set(xticks=x, xticklabels=frame.short, ylabel="Samples", title="A  HPLC measurement eligibility")
    ax.set_ylim(0, float(frame.n_total.max()) * 1.22)
    ax.legend(fontsize=8, ncol=3, loc="upper center")
    ax = axes[0, 1]
    frame = load(data_dir, "Figure_2B_endpoint_denominators.csv")
    order = frame.sort_values("plot_order").short.drop_duplicates().tolist()
    x = np.arange(len(order))
    for k, (kind, color) in enumerate([("Portions", TEAL), ("Batches", ORANGE)]):
        part = frame.loc[frame.denominator.eq(kind)].set_index("short").loc[order]
        ax.bar(x + (k - .5) * .34, part.n, width=.32, label=kind, color=color)
    ax.set(xticks=x, xticklabels=order, ylabel="Eligible count", title="B  Endpoint denominators")
    ax.set_ylim(0, float(frame.n.max()) * 1.20)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax = axes[1, 0]
    frame = load(data_dir, "Figure_2C_heterogeneity_summary.csv").sort_values("plot_order")
    forest(ax, frame, "median_normalized_range_global_iqr", "ci_low_normalized_range_global_iqr", "ci_high_normalized_range_global_iqr")
    ax.set(yticks=np.arange(len(frame)), yticklabels=frame.short, xlabel="Within-batch range / analyte IQR", title="C  Within-batch heterogeneity (95% CI)")
    ax.invert_yaxis()
    ax = axes[1, 1]
    frame = load(data_dir, "Figure_2D_appearance_chemistry_effects.csv")
    for k, (anchor, label, color) in enumerate([
        ("expert_appearance_grade", "Recorded grade", ORANGE),
        ("cross_fitted_visible_score", "OOF visible score", TEAL),
    ]):
        part = frame.loc[frame.anchor_id.eq(anchor)].set_index("short").loc[ANALYTES]
        forest(ax, part, "spearman_rho", "ci_low", "ci_high", y=np.arange(7) + (k - .5) * .25, color=color, label=label)
    ax.set(yticks=np.arange(7), yticklabels=ANALYTES, xlabel="Spearman correlation (95% CI)", title="D  Appearance–chemistry association")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    save(fig, output_dir, "Figure_2")


def figure3(data_dir, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    ax = axes[0]
    frame = load(data_dir, "Figure_3D_OOF_confusion_matrix.csv")
    counts = frame.pivot(index="true_grade", columns="predicted_grade", values="n").reindex(index=[1, 2, 3], columns=[1, 2, 3])
    fractions = frame.pivot(index="true_grade", columns="predicted_grade", values="row_fraction").reindex_like(counts)
    ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, f"{counts.iloc[row, col]:.0f}\n{100 * fractions.iloc[row, col]:.0f}%", ha="center", va="center", color="white" if fractions.iloc[row, col] > .5 else "#253746")
    ax.set(xticks=range(3), xticklabels=[1, 2, 3], yticks=range(3), yticklabels=[1, 2, 3], xlabel="Predicted grade", ylabel="Recorded grade", title="D  OOF confusion matrix")
    ax = axes[1]
    frame = load(data_dir, "Figure_3E_visible_grade_metrics.csv")
    forest(ax, frame, "estimate", "ci_low", "ci_high")
    ax.set(yticks=np.arange(len(frame)), yticklabels=frame.metric, xlim=(0, 1), xlabel="F1 (95% CI)", title="E  Visible classification")
    ax.invert_yaxis()
    ax = axes[2]
    frame = load(data_dir, "Figure_3F_repeat_agreement_summary.csv")
    forest(ax, frame, "observed_accuracy", "ci_low", "ci_high")
    ax.set(yticks=np.arange(len(frame)), yticklabels=[f"{r.agreement_bin} (n={r.n_portions})" for r in frame.itertuples()], xlim=(0, 1.03), xlabel="Observed accuracy (95% CI)", ylabel="Agreement over 10 predictions", title="F  Prediction agreement")
    ax.invert_yaxis()
    save(fig, output_dir, "Figure_3_D_to_F")


def figure4(data_dir, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, name, value, low, high, title, xlabel in [
        (axes[0, 0], "Figure_4A_HSI_P0_OOF_R2.csv", "estimate", "ci_low", "ci_high", "A  HSI P0 prediction", "OOF R² (95% CI)"),
        (axes[0, 1], "Figure_4B_HSI_IQR_normalized_RMSE.csv", "nrmse_iqr", "nrmse_iqr_low", "nrmse_iqr_high", "B  IQR-normalized prediction error", "RMSE / IQR (95% CI)"),
    ]:
        frame = load(data_dir, name).set_index("short").loc[ANALYTES]
        forest(ax, frame, value, low, high)
        ax.set(yticks=range(7), yticklabels=ANALYTES, xlabel=xlabel, title=title)
        ax.invert_yaxis()
    frame = load(data_dir, "Figure_4C_endpoint_conditional_increment.csv")
    contexts = ["Beyond domain", "Beyond appearance", "Beyond both"]
    ax = axes[1, 0]
    for k, (context, color) in enumerate(zip(contexts, [TEAL, ORANGE, BLUE])):
        part = frame.loc[frame.context.eq(context)].set_index("short").loc[ANALYTES]
        y = np.arange(7) + (k - 1) * .2
        ax.scatter(part.relative_mse_reduction * 100, y, color=color, label=context, s=24)
        for position, row in zip(y, part.itertuples()):
            if row.q_bh < .05:
                ax.annotate("*", (row.relative_mse_reduction * 100, position), xytext=(4, 0), textcoords="offset points", va="center", color=color)
    ax.axvline(0, color="#B9C1C5", linewidth=.8)
    ax.set(yticks=range(7), yticklabels=ANALYTES, xlabel="Relative MSE reduction (%)", title="C  Conditional increments (* BH q < 0.05)")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    profiles = load(data_dir, "Figure_4D_evidence_profiles.csv").set_index("short").loc[ANALYTES]
    effects = frame.pivot(index="short", columns="context", values="relative_mse_reduction").loc[ANALYTES, contexts]
    supported = (profiles[contexts].to_numpy(float) < .05) & (effects.to_numpy(float) > 0)
    ax.imshow(supported, cmap=ListedColormap(["#E9EFF0", TEAL]), vmin=0, vmax=1, aspect="auto")
    for row in range(7):
        for col in range(3):
            ax.text(col, row, "Yes" if supported[row, col] else "No", ha="center", va="center", color="white" if supported[row, col] else "#495960", fontsize=8)
    ax.set(xticks=range(3), xticklabels=["Domain", "Appearance", "Both"], yticks=range(7), yticklabels=[f"{name}: {profiles.loc[name, 'evidence_class']}" for name in ANALYTES], title="D  Evidence profiles", xlabel="Supported increment beyond baseline")
    ax.tick_params(axis="y", labelsize=7)
    save(fig, output_dir, "Figure_4")


def session_matrix(frame):
    """Return all 49 percentages using held-session baseline MSE denominators."""
    sessions = [f"S{i}" for i in range(1, 8)]
    keys = pd.MultiIndex.from_product([ANALYTES, sessions], names=["short", "held_session"])
    if len(frame) != 49 or frame.duplicated(["short", "held_session"]).any():
        raise ValueError("Figure 5D requires exactly 49 unique analyte–session rows")
    ordered = frame.set_index(["short", "held_session"]).reindex(keys)
    baseline = ordered.baseline_mse.to_numpy(float)
    hsi = ordered.hsi_mse.to_numpy(float)
    if not np.isfinite(baseline).all() or not np.isfinite(hsi).all() or np.any(baseline <= 0):
        raise ValueError("Figure 5D requires finite errors and positive session baseline MSE")
    values = 100 * (baseline - hsi) / baseline
    np.testing.assert_allclose(values, ordered.percentage_mse_reduction, rtol=1e-10, atol=1e-10)
    return values.reshape(7, 7), sessions


def figure5(data_dir, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4), layout="constrained")
    frame = load(data_dir, "Figure_5AB_multianalyte_panel_effects.csv")
    comparisons = frame.comparison_id.drop_duplicates().tolist()
    labels = ["Mean baseline", "Domain baseline", "Appearance baseline", "Domain + appearance"]
    for ax, both, title in [(axes[0, 0], False, "A  Primary seven-analyte panel"), (axes[0, 1], True, "B  Panel sensitivity")]:
        for k, (panel, label, color) in enumerate([("all_seven", "Seven analytes", TEAL), ("six_without_catechin", "Six without CAT (post hoc)", ORANGE)]):
            if k and not both:
                continue
            part = frame.loc[frame.panel_id.eq(panel)].set_index("comparison_id").loc[comparisons]
            forest(ax, part, "endpoint_batch_equal_error_reduction", "ci_low", "ci_high", y=np.arange(4) + ((k - .5) * .22 if both else 0), color=color, scale=100, label=label)
        ax.set(yticks=range(4), yticklabels=labels, xlabel="Relative MSE reduction (%, 95% CI)", title=title)
        ax.invert_yaxis()
        if both:
            ax.legend(fontsize=8, loc="center right")
    ax = axes[1, 0]
    frame = load(data_dir, "Figure_5C_grouped_CV_vs_LOSO_R2.csv")
    for k, (validation, color) in enumerate(zip(frame.validation.drop_duplicates(), [TEAL, ORANGE])):
        part = frame.loc[frame.validation.eq(validation)].set_index("short").loc[ANALYTES]
        forest(ax, part, "estimate", "ci_low", "ci_high", y=np.arange(7) + (k - .5) * .24, color=color, label=validation)
    ax.set(yticks=range(7), yticklabels=ANALYTES, xlabel="R² (95% CI)", title="C  Grouped CV and cross-session performance")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    matrix, sessions = session_matrix(load(data_dir, "Figure_5D_session_transport_heatmap.csv"))
    norm = TwoSlopeNorm(vmin=min(float(matrix.min()), -1), vcenter=0, vmax=max(float(matrix.max()), 1))
    heat = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
    for row in range(7):
        for col in range(7):
            scaled = norm(matrix[row, col])
            ax.text(col, row, f"{matrix[row, col]:.0f}", ha="center", va="center", color="white" if scaled < .22 or scaled > .8 else "#243746", fontsize=8)
    ax.set(xticks=range(7), xticklabels=sessions, yticks=range(7), yticklabels=ANALYTES, xlabel="Held-out acquisition session", title="D  Session-specific relative MSE reduction")
    fig.colorbar(heat, ax=ax, label="Relative MSE reduction (%)", shrink=.85)
    save(fig, output_dir, "Figure_5")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "paper_results/figure_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/figures")
    args = parser.parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if output_dir == data_dir or data_dir in output_dir.parents:
        raise SystemExit("Use a separate output directory to preserve source CSVs.")
    # Validate all required inputs before writing any output.
    required = ["Figure_2A_HPLC_range_qualification.csv", "Figure_2B_endpoint_denominators.csv", "Figure_2C_heterogeneity_summary.csv", "Figure_2D_appearance_chemistry_effects.csv", "Figure_3D_OOF_confusion_matrix.csv", "Figure_3E_visible_grade_metrics.csv", "Figure_3F_repeat_agreement_summary.csv", "Figure_4A_HSI_P0_OOF_R2.csv", "Figure_4B_HSI_IQR_normalized_RMSE.csv", "Figure_4C_endpoint_conditional_increment.csv", "Figure_4D_evidence_profiles.csv", "Figure_5AB_multianalyte_panel_effects.csv", "Figure_5C_grouped_CV_vs_LOSO_R2.csv", "Figure_5D_session_transport_heatmap.csv"]
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit("Missing public figure CSVs: " + ", ".join(missing))
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    for build in (figure2, figure3, figure4, figure5):
        build(data_dir, output_dir)
    print("Created PNG/PDF panels: Figure 2, Figure 3D–F, Figure 4 and Figure 5.")


if __name__ == "__main__":
    main()
