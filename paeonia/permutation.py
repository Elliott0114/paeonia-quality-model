"""Batch-label permutation sensitivity for the fixed appearance classifier."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline


VISIBLE15 = [
    "machine_average_area_cm2",
    "machine_short_axis_lt_0p8_pct",
    "machine_abnormal_piece_pct",
    "visible_mask_component_area_cv",
    "visible_lab_L_mean",
    "visible_lab_a_mean",
    "visible_lab_b_mean",
    "visible_color_delta_e_fraction_gt_12",
    "visible_object_L_between_std",
    "visible_relative_dark_spot_area_fraction",
    "visible_edge_minus_inner_L_mean",
    "visible_texture_local_L_std_w5_mean",
    "visible_texture_local_L_std_w33_mean",
    "visible_texture_coarse_to_fine_ratio",
    "visible_glcm_entropy",
]


def estimator(seed: int) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            l2_regularization=0.1,
            random_state=seed,
        ),
    )


def build_split_batches(predictions: pd.DataFrame) -> list[dict[str, object]]:
    fixed = predictions.loc[predictions["analysis_model"].eq("nested_selected")].copy()
    rows: list[dict[str, object]] = []
    for (repeat, outer_fold), group in fixed.groupby(["repeat", "outer_fold"], observed=True):
        rows.append(
            {
                "repeat": int(repeat),
                "outer_fold": int(outer_fold),
                "test_batches": tuple(sorted(group["sample_number"].unique())),
            }
        )
    if len(rows) != 50:
        raise ValueError(f"Expected 50 outer splits; found {len(rows)}")
    return rows


def evaluate_labels(
    table: pd.DataFrame,
    split_batches: list[dict[str, object]],
    batch_labels: dict[int, int],
    seed: int,
) -> tuple[float, list[float]]:
    x = table[VISIBLE15].to_numpy(dtype=float)
    groups = table["sample_number"].to_numpy(dtype=int)
    y = np.array([batch_labels[int(group)] for group in groups], dtype=int)
    prediction_rows: list[tuple[int, int, int]] = []

    for split in split_batches:
        test_mask = np.isin(groups, split["test_batches"])
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        model_seed = seed + int(split["repeat"]) * 1009 + int(split["outer_fold"])
        model = estimator(model_seed)
        model.fit(x[train], y[train])
        prediction = model.predict(x[test])
        prediction_rows.extend(
            (int(split["repeat"]), int(index), int(value))
            for index, value in zip(test, prediction, strict=True)
        )

    predictions = pd.DataFrame(
        prediction_rows,
        columns=["repeat", "row_index", "prediction"],
    )
    repeat_scores: list[float] = []
    for _, group in predictions.groupby("repeat", observed=True):
        if len(group) != 207 or group["row_index"].nunique() != 207:
            raise RuntimeError("Each repeat must contain one prediction per portion")
        index = group["row_index"].to_numpy(dtype=int)
        repeat_scores.append(
            float(f1_score(y[index], group["prediction"], average="macro", zero_division=0))
        )
    return float(np.mean(repeat_scores)), repeat_scores


def permutation_task(
    permutation: int,
    table: pd.DataFrame,
    split_batches: list[dict[str, object]],
    batch_ids: np.ndarray,
    original_labels: np.ndarray,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + permutation * 7919)
    permuted = rng.permutation(original_labels)
    labels = {int(batch): int(label) for batch, label in zip(batch_ids, permuted, strict=True)}
    statistic, repeat_scores = evaluate_labels(
        table,
        split_batches,
        labels,
        seed + permutation * 104729,
    )
    return {
        "permutation": permutation,
        "mean_repeated_macro_f1": statistic,
        "minimum_repeat_macro_f1": min(repeat_scores),
        "maximum_repeat_macro_f1": max(repeat_scores),
    }
