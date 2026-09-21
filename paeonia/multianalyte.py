"""Seven-analyte cross-modal comparisons and equal-weight panels."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from . import cross_modal as crossmodal
from . import chemical as final


COMPOUNDS = (
    "没食子酸",
    "氧化芍药苷",
    "儿茶素",
    "芍药内酯苷",
    "芍药苷",
    "没食子酰芍药苷",
    "苯甲酰芍药苷",
)


PANELS = {
    "all_seven": COMPOUNDS,
    "six_without_catechin": tuple(item for item in COMPOUNDS if item != final.CATECHIN),
}


COMPARISONS = (
    (
        "deployable_hsi_beyond_visible15",
        "visible15_ridge",
        "visible15_plus_hsi_P0",
        True,
    ),
    (
        "confounder_aware_hsi_beyond_visible15",
        "domain_plus_visible15",
        "domain_plus_visible15_plus_hsi_residual",
        True,
    ),
    (
        "descriptive_visible15_beyond_hsi",
        "hsi_P0",
        "visible15_plus_hsi_P0",
        False,
    ),
)


def run_split_endpoint(
    split: final.CVSplit,
    compound: str,
    table: pd.DataFrame,
    visible: np.ndarray,
    spectra: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    test = table.iloc[split.test_idx]
    for model_id in crossmodal.MODEL_IDS:
        alpha: float | None = None
        if model_id in crossmodal.TUNED_MODEL_IDS:
            alpha, scores = crossmodal.select_alpha(
                model_id,
                table,
                visible,
                spectra,
                y,
                split.inner_splits,
                alphas,
            )
            for row in scores.to_dict(orient="records"):
                selection_rows.append(
                    {
                        "compound": compound,
                        "split_id": split.split_id,
                        "repeat": split.repeat,
                        "outer_fold": split.fold,
                        "model_id": model_id,
                        **row,
                    }
                )
        prediction = crossmodal.fit_predict(
            model_id,
            table,
            visible,
            spectra,
            y,
            split.train_idx,
            split.test_idx,
            alpha,
        )
        for local_index, (_, row) in enumerate(test.iterrows()):
            global_index = split.test_idx[local_index]
            observed = bool(np.isfinite(y[global_index]))
            prediction_rows.append(
                {
                    "compound": compound,
                    "split_id": split.split_id,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "canonical_id": row["canonical_id"],
                    "sample_number": int(row["sample_number"]),
                    "group_index": int(row["group_index"]),
                    "session_code": row["session_code"],
                    "hplc_calibration_group": int(row["hplc_calibration_group"]),
                    "model_id": model_id,
                    "selected_alpha": alpha,
                    "observed_target": observed,
                    "y_true": float(y[global_index]) if observed else math.nan,
                    "y_pred": float(prediction[local_index]),
                }
            )
    return prediction_rows, selection_rows


def holm_adjust(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    output = np.full(len(array), np.nan)
    valid = np.flatnonzero(np.isfinite(array))
    if not len(valid):
        return pd.Series(output, index=values.index)
    order = valid[np.argsort(array[valid])]
    adjusted = array[order] * np.arange(len(order), 0, -1)
    adjusted = np.maximum.accumulate(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return pd.Series(output, index=values.index)


def paired_errors(
    predictions: pd.DataFrame,
    baseline_model: str,
    augmented_model: str,
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["observed_target"]
        & predictions["model_id"].isin([baseline_model, augmented_model])
    ].copy()
    keys = ["compound", "repeat", "canonical_id", "sample_number"]
    baseline = subset.loc[
        subset["model_id"].eq(baseline_model), [*keys, "y_true", "y_pred"]
    ].rename(columns={"y_true": "y_true_baseline", "y_pred": "y_pred_baseline"})
    augmented = subset.loc[
        subset["model_id"].eq(augmented_model), [*keys, "y_true", "y_pred"]
    ].rename(columns={"y_true": "y_true_augmented", "y_pred": "y_pred_augmented"})
    output = baseline.merge(
        augmented,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if not np.allclose(output["y_true_baseline"], output["y_true_augmented"]):
        raise ValueError("Paired targets do not match")
    output["y_true"] = output.pop("y_true_baseline")
    output = output.drop(columns="y_true_augmented")
    output["baseline_squared_error"] = np.square(
        output["y_true"] - output["y_pred_baseline"]
    )
    output["augmented_squared_error"] = np.square(
        output["y_true"] - output["y_pred_augmented"]
    )
    output["squared_error_reduction"] = (
        output["baseline_squared_error"] - output["augmented_squared_error"]
    )
    return output[
        [
            "compound",
            "repeat",
            "canonical_id",
            "sample_number",
            "y_true",
            "baseline_squared_error",
            "augmented_squared_error",
            "squared_error_reduction",
        ]
    ].copy()


def panel_summary(
    paired: pd.DataFrame,
    comparison_id: str,
    panel_id: str,
    panel: tuple[str, ...],
    normalization: str,
    signflips: int,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    batch_sets = [
        set(paired.loc[paired["compound"].eq(compound), "sample_number"].unique())
        for compound in panel
    ]
    batches = sorted(set.intersection(*batch_sets))
    subset = paired.loc[
        paired["compound"].isin(panel) & paired["sample_number"].isin(batches)
    ].copy()
    scales: dict[str, float] = {}
    scale_rows: list[dict[str, Any]] = []
    for compound in panel:
        group = subset.loc[subset["compound"].eq(compound)]
        baseline_mse = float(group["baseline_squared_error"].mean())
        targets = group[["canonical_id", "y_true"]].drop_duplicates()
        target_variance = float(targets["y_true"].var(ddof=0))
        scale = baseline_mse if normalization == "baseline_mse" else target_variance
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid {normalization} scale for {compound}: {scale}")
        scales[compound] = scale
        scale_rows.append(
            {
                "comparison_id": comparison_id,
                "panel_id": panel_id,
                "compound": compound,
                "normalization": normalization,
                "scale": scale,
                "baseline_mse": baseline_mse,
                "target_variance": target_variance,
                "n_portions": int(targets["canonical_id"].nunique()),
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
        index="sample_number",
        columns="compound",
        values="normalized_error_reduction",
    ).reindex(columns=list(panel))
    if pivot.isna().any().any():
        raise ValueError("Incomplete batch-by-endpoint panel score")
    scores = pivot.mean(axis=1).rename("panel_score").reset_index()
    scores.insert(0, "comparison_id", comparison_id)
    scores.insert(1, "panel_id", panel_id)
    scores.insert(2, "normalization", normalization)
    effect = float(scores["panel_score"].mean())
    rng = np.random.default_rng(final.stable_seed(seed, comparison_id, panel_id, normalization))
    values = scores["panel_score"].to_numpy(dtype=float)
    draws = rng.choice(values, size=(bootstraps, len(values)), replace=True).mean(axis=1)
    result = {
        "comparison_id": comparison_id,
        "panel_id": panel_id,
        "included_compounds": ";".join(panel),
        "n_endpoints": len(panel),
        "normalization": normalization,
        "n_batches": len(batches),
        "n_prediction_rows": int(len(subset)),
        "n_unique_portions_union": int(subset["canonical_id"].nunique()),
        "endpoint_batch_equal_error_reduction": effect,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "signflip_p": final.signflip_pvalue(
            values,
            signflips,
            final.stable_seed(seed, comparison_id, panel_id, normalization, "signflip"),
        ),
        "signflip_replicates": signflips,
        "bootstrap_replicates": bootstraps,
    }
    return result, scores, pd.DataFrame(scale_rows)
