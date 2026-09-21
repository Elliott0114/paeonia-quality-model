"""Mean/domain panel effects with fixed normalization denominators."""
from __future__ import annotations

import hashlib
import math
from typing import Any
import numpy as np
import pandas as pd


COMPOUND_ORDER = (
    "没食子酸",
    "氧化芍药苷",
    "儿茶素",
    "芍药内酯苷",
    "芍药苷",
    "没食子酰芍药苷",
    "苯甲酰芍药苷",
)


COMPOUND_SHORT = {
    "没食子酸": "GA",
    "氧化芍药苷": "OPF",
    "儿茶素": "CAT",
    "芍药内酯苷": "ALB",
    "芍药苷": "PF",
    "没食子酰芍药苷": "GPF",
    "没食子酰葡萄糖": "PGG",
    "苯甲酰芍药苷": "BPF",
}


COMPARISONS = {
    "absolute_hsi_gain": ("mean_train", "hsi_P0"),
    "conditional_hsi_gain": ("domain_only", "domain_plus_hsi_orthogonal"),
}


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = valid[np.argsort(values[valid])]
    ranked = values[order] * np.arange(len(order), 0, -1)
    ranked = np.maximum.accumulate(ranked)
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def signflip_pvalue(values: np.ndarray, replicates: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    while completed < replicates:
        size = min(10_000, replicates - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        null = np.mean(signs * values, axis=1)
        exceed += int(np.sum(null >= observed))
        completed += size
    return float((exceed + 1) / (replicates + 1))


def bootstrap_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def pair_predictions(
    predictions: pd.DataFrame,
    baseline_model: str,
    augmented_model: str,
) -> pd.DataFrame:
    keys = ["compound", "repeat", "canonical_id", "sample_number"]
    columns = [*keys, "y_true", "y_pred"]
    baseline = predictions.loc[predictions["model_id"].eq(baseline_model), columns].rename(
        columns={"y_true": "y_true_baseline", "y_pred": "y_pred_baseline"}
    )
    augmented = predictions.loc[predictions["model_id"].eq(augmented_model), columns].rename(
        columns={"y_true": "y_true_augmented", "y_pred": "y_pred_augmented"}
    )
    paired = baseline.merge(augmented, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(
        paired["y_true_baseline"], paired["y_true_augmented"], equal_nan=True
    ):
        raise ValueError("Paired baseline and augmented targets do not match")
    paired["y_true"] = paired.pop("y_true_baseline")
    paired = paired.drop(columns="y_true_augmented")
    finite = np.isfinite(
        paired[["y_true", "y_pred_baseline", "y_pred_augmented"]].to_numpy(dtype=float)
    ).all(axis=1)
    paired = paired.loc[finite].copy()
    paired["baseline_squared_error"] = np.square(
        paired["y_true"] - paired["y_pred_baseline"]
    )
    paired["augmented_squared_error"] = np.square(
        paired["y_true"] - paired["y_pred_augmented"]
    )
    paired["squared_error_reduction"] = (
        paired["baseline_squared_error"] - paired["augmented_squared_error"]
    )
    return paired


def common_panel_batches(paired: pd.DataFrame, panel: tuple[str, ...]) -> list[int]:
    batch_sets = [
        set(paired.loc[paired["compound"].eq(compound), "sample_number"].unique())
        for compound in panel
    ]
    return sorted(set.intersection(*batch_sets))


def panel_scores(
    paired: pd.DataFrame,
    panel: tuple[str, ...],
    normalization: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    batches = common_panel_batches(paired, panel)
    subset = paired.loc[
        paired["compound"].isin(panel) & paired["sample_number"].isin(batches)
    ].copy()
    scales: dict[str, float] = {}
    scale_rows: list[dict[str, Any]] = []
    for compound in panel:
        group = subset.loc[subset["compound"].eq(compound)]
        baseline_mse = float(group["baseline_squared_error"].mean())
        unique_targets = group[["canonical_id", "y_true"]].drop_duplicates()
        target_variance = float(unique_targets["y_true"].var(ddof=0))
        scale = baseline_mse if normalization == "baseline_mse" else target_variance
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid {normalization} scale for {compound}: {scale}")
        scales[compound] = scale
        scale_rows.append(
            {
                "compound": compound,
                "compound_short": COMPOUND_SHORT[compound],
                "normalization": normalization,
                "scale": scale,
                "baseline_mse": baseline_mse,
                "target_variance": target_variance,
                "n_portions": int(unique_targets["canonical_id"].nunique()),
                "n_batches": int(group["sample_number"].nunique()),
            }
        )
    batch_analyte = (
        subset.groupby(["sample_number", "compound"], observed=True)[
            "squared_error_reduction"
        ]
        .mean()
        .rename("mean_squared_error_reduction")
        .reset_index()
    )
    batch_analyte["normalization_scale"] = batch_analyte["compound"].map(scales)
    batch_analyte["normalized_error_reduction"] = (
        batch_analyte["mean_squared_error_reduction"]
        / batch_analyte["normalization_scale"]
    )
    pivot = batch_analyte.pivot(
        index="sample_number", columns="compound", values="normalized_error_reduction"
    ).reindex(columns=list(panel))
    if pivot.isna().any().any():
        raise ValueError("Primary panel cohort has incomplete batch-by-analyte scores")
    scores = pivot.mean(axis=1).rename("panel_score").reset_index()
    counts = {
        "n_batches": len(batches),
        "n_prediction_rows": int(len(subset)),
        "n_unique_portions_union": int(subset["canonical_id"].nunique()),
        "min_endpoint_portions": int(
            subset.groupby("compound", observed=True)["canonical_id"].nunique().min()
        ),
        "max_endpoint_portions": int(
            subset.groupby("compound", observed=True)["canonical_id"].nunique().max()
        ),
    }
    return scores, pd.DataFrame(scale_rows), counts


def summarize_panel(
    paired: pd.DataFrame,
    comparison_id: str,
    baseline_model: str,
    augmented_model: str,
    panel_id: str,
    panel: tuple[str, ...],
    normalization: str,
    signflips: int,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    scores, scales, counts = panel_scores(paired, panel, normalization)
    values = scores["panel_score"].to_numpy(dtype=float)
    low, high = bootstrap_interval(
        values,
        bootstraps,
        stable_seed(seed, comparison_id, panel_id, normalization, "bootstrap"),
    )
    row = {
        "comparison_id": comparison_id,
        "baseline_model": baseline_model,
        "augmented_model": augmented_model,
        "panel_id": panel_id,
        "included_compounds": ";".join(panel),
        "n_endpoints": len(panel),
        "normalization": normalization,
        **counts,
        "endpoint_batch_equal_error_reduction": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "signflip_p": signflip_pvalue(
            values,
            signflips,
            stable_seed(seed, comparison_id, panel_id, normalization, "signflip"),
        ),
        "signflip_replicates": signflips,
        "bootstrap_replicates": bootstraps,
    }
    scores.insert(0, "normalization", normalization)
    scores.insert(0, "panel_id", panel_id)
    scores.insert(0, "comparison_id", comparison_id)
    scales.insert(0, "panel_id", panel_id)
    scales.insert(0, "comparison_id", comparison_id)
    return row, scores, scales
