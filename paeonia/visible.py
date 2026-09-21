"""Batch-grouped appearance classification with the fixed 15-feature panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


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


def estimators(seed: int) -> dict[str, object]:
    return {
        "logreg_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "linear_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LinearSVC(
                class_weight="balanced",
                random_state=seed,
                dual="auto",
                max_iter=20000,
            ),
        ),
        "rbf_svm": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=3.0,
                gamma="scale",
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=500,
                random_state=seed,
                class_weight="balanced",
                min_samples_leaf=2,
                n_jobs=1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=250,
                l2_regularization=0.1,
                random_state=seed,
            ),
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
        "mean_absolute_grade_error": float(mean_absolute_error(y_true, y_pred)),
    }


def grouped_splits(
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = []
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("Product-batch leakage detected")
        result.append((train, test))
    return result


def select_model(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    candidates = estimators(seed)
    splits = grouped_splits(y, groups, folds, seed)
    for name, prototype in candidates.items():
        fold_scores = []
        for train, valid in splits:
            model = clone(prototype)
            model.fit(x[train], y[train])
            prediction = model.predict(x[valid])
            fold_scores.append(f1_score(y[valid], prediction, average="macro", zero_division=0))
        rows.append(
            {
                "candidate_model": name,
                "inner_macro_f1_mean": float(np.mean(fold_scores)),
                "inner_macro_f1_sd": float(np.std(fold_scores, ddof=1)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["inner_macro_f1_mean", "candidate_model"], ascending=[False, True]
    )
    return str(scores.iloc[0]["candidate_model"]), scores


def cluster_bootstrap(
    frame: pd.DataFrame,
    cluster: str,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    clusters = frame[cluster].drop_duplicates().to_numpy()
    cluster_rows = {
        value: frame.loc[frame[cluster].eq(value), ["true_grade", "predicted_grade"]]
        for value in clusters
    }
    draws: list[dict[str, float]] = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([cluster_rows[value] for value in sampled], ignore_index=True)
        if sample["true_grade"].nunique() < 3:
            continue
        draws.append(metrics(sample["true_grade"].to_numpy(), sample["predicted_grade"].to_numpy()))
    draw_frame = pd.DataFrame(draws)
    point = metrics(frame["true_grade"].to_numpy(), frame["predicted_grade"].to_numpy())
    return pd.DataFrame(
        [
            {
                "metric": name,
                "estimate": value,
                "ci_low": float(draw_frame[name].quantile(0.025)),
                "ci_high": float(draw_frame[name].quantile(0.975)),
                "bootstrap_valid_replicates": int(draw_frame[name].notna().sum()),
                "ci_type": "product_batch_cluster_bootstrap",
            }
            for name, value in point.items()
        ]
    )
