"""Inner-session tuning and session-equal leave-one-session-out inference."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class LOSOSplit:
    split_id: str
    repeat: int
    fold: int
    held_session: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_splits: tuple[tuple[np.ndarray, np.ndarray], ...]


def build_loso_splits(table: pd.DataFrame, smoke: bool) -> list[LOSOSplit]:
    sessions = sorted(table["session_code"].unique())
    if smoke:
        sessions = sessions[:2]
    output: list[LOSOSplit] = []
    for fold, session in enumerate(sessions, start=1):
        test_idx = np.flatnonzero(table["session_code"].eq(session).to_numpy())
        train_idx = np.flatnonzero(~table["session_code"].eq(session).to_numpy())
        remaining = sorted(table.iloc[train_idx]["session_code"].unique())
        if smoke:
            remaining = remaining[:3]
        inner: list[tuple[np.ndarray, np.ndarray]] = []
        for inner_session in remaining:
            inner_valid = train_idx[
                table.iloc[train_idx]["session_code"].eq(inner_session).to_numpy()
            ]
            inner_train = np.setdiff1d(train_idx, inner_valid, assume_unique=True)
            inner.append((inner_train, inner_valid))
        output.append(
            LOSOSplit(
                split_id=f"LOSO_{session}",
                repeat=1,
                fold=fold,
                held_session=str(session),
                train_idx=train_idx,
                test_idx=test_idx,
                inner_splits=tuple(inner),
            )
        )
    return output


def run_split_endpoint(
    split: LOSOSplit,
    compound: str,
    table: pd.DataFrame,
    visible: np.ndarray,
    spectra: np.ndarray,
    y: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alpha, scores = crossmodal.select_alpha(
        "hsi_P0",
        table,
        visible,
        spectra,
        y,
        split.inner_splits,
        alphas,
    )
    hsi_prediction = crossmodal.fit_predict(
        "hsi_P0",
        table,
        visible,
        spectra,
        y,
        split.train_idx,
        split.test_idx,
        alpha,
    )
    observed_train = split.train_idx[np.isfinite(y[split.train_idx])]
    mean_prediction = np.full(len(split.test_idx), float(np.mean(y[observed_train])))
    rows: list[dict[str, Any]] = []
    test = table.iloc[split.test_idx]
    for model_id, prediction, selected_alpha in (
        ("mean_train", mean_prediction, math.nan),
        ("hsi_P0", hsi_prediction, alpha),
    ):
        for local_index, (_, row) in enumerate(test.iterrows()):
            global_index = split.test_idx[local_index]
            observed = bool(np.isfinite(y[global_index]))
            rows.append(
                {
                    "compound": compound,
                    "split_id": split.split_id,
                    "repeat": 1,
                    "outer_fold": split.fold,
                    "held_session": split.held_session,
                    "canonical_id": row["canonical_id"],
                    "sample_number": int(row["sample_number"]),
                    "group_index": int(row["group_index"]),
                    "session_code": row["session_code"],
                    "hplc_calibration_group": int(row["hplc_calibration_group"]),
                    "model_id": model_id,
                    "selected_alpha": selected_alpha,
                    "observed_target": observed,
                    "y_true": float(y[global_index]) if observed else math.nan,
                    "y_pred": float(prediction[local_index]),
                }
            )
    selection_rows = [
        {
            "compound": compound,
            "split_id": split.split_id,
            "outer_fold": split.fold,
            "held_session": split.held_session,
            "model_id": "hsi_P0",
            **row,
        }
        for row in scores.to_dict(orient="records")
    ]
    return rows, selection_rows


def exact_signflip_pvalue(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(np.mean(values))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
    null = np.mean(signs * values, axis=1)
    return float(np.mean(null >= observed - 1e-15))


def bootstrap_interval(values: np.ndarray, replicates: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.loc[
        predictions["observed_target"]
        & predictions["model_id"].isin(["mean_train", "hsi_P0"])
    ].copy()
    keys = [
        "compound",
        "held_session",
        "canonical_id",
        "sample_number",
    ]
    baseline = scored.loc[
        scored["model_id"].eq("mean_train"), [*keys, "y_true", "y_pred"]
    ].rename(columns={"y_true": "y_true_baseline", "y_pred": "y_pred_baseline"})
    hsi = scored.loc[
        scored["model_id"].eq("hsi_P0"), [*keys, "y_true", "y_pred"]
    ].rename(columns={"y_true": "y_true_hsi", "y_pred": "y_pred_hsi"})
    paired = baseline.merge(hsi, on=keys, how="inner", validate="one_to_one")
    if not np.allclose(paired["y_true_baseline"], paired["y_true_hsi"]):
        raise ValueError("LOSO paired targets do not match")
    paired["y_true"] = paired.pop("y_true_baseline")
    paired = paired.drop(columns="y_true_hsi")
    paired["baseline_squared_error"] = np.square(
        paired["y_true"] - paired["y_pred_baseline"]
    )
    paired["hsi_squared_error"] = np.square(paired["y_true"] - paired["y_pred_hsi"])
    paired["squared_error_reduction"] = (
        paired["baseline_squared_error"] - paired["hsi_squared_error"]
    )
    return paired


def endpoint_increment_results(
    paired: pd.DataFrame,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    session_rows: list[pd.DataFrame] = []
    for compound, group in paired.groupby("compound", observed=True):
        by_session = (
            group.groupby("held_session", observed=True)
            .agg(
                n_portions=("canonical_id", "size"),
                n_batches=("sample_number", "nunique"),
                mean_squared_error_reduction=("squared_error_reduction", "mean"),
                baseline_mse=("baseline_squared_error", "mean"),
                hsi_mse=("hsi_squared_error", "mean"),
            )
            .reset_index()
        )
        by_session.insert(0, "compound", compound)
        session_baseline = by_session["baseline_mse"].to_numpy(dtype=float)
        if not np.all(np.isfinite(session_baseline) & (session_baseline > 0)):
            raise ValueError(f"Invalid held-session baseline MSE for {compound}")
        by_session["relative_mse_reduction"] = (
            by_session["mean_squared_error_reduction"] / by_session["baseline_mse"]
        )
        by_session["percentage_mse_reduction"] = 100 * by_session["relative_mse_reduction"]
        session_rows.append(by_session)
        values = by_session["mean_squared_error_reduction"].to_numpy(dtype=float)
        low, high = bootstrap_interval(
            values,
            bootstraps,
            final.stable_seed(seed, compound, "session_bootstrap"),
        )
        baseline_mse = float(group["baseline_squared_error"].mean())
        hsi_mse = float(group["hsi_squared_error"].mean())
        rows.append(
            {
                "compound": compound,
                "n_portions": int(group["canonical_id"].nunique()),
                "n_batches": int(group["sample_number"].nunique()),
                "n_sessions": int(by_session["held_session"].nunique()),
                "baseline_mse": baseline_mse,
                "hsi_mse": hsi_mse,
                "row_weighted_relative_mse_reduction": (
                    (baseline_mse - hsi_mse) / baseline_mse
                ),
                "session_equal_mse_reduction": float(np.mean(values)),
                # Fixed pooled endpoint denominator; distinct from averaging session percentages.
                "session_equal_normalized_mse_reduction": float(np.mean(values)) / baseline_mse,
                "normalized_ci_low": low / baseline_mse,
                "normalized_ci_high": high / baseline_mse,
                "ci_low": low,
                "ci_high": high,
                "exact_signflip_p": exact_signflip_pvalue(values),
                "sign_assignments": 2 ** len(values),
            }
        )
    result = pd.DataFrame(rows)
    result["signflip_q_bh"] = final.benjamini_hochberg(result["exact_signflip_p"])
    return result, pd.concat(session_rows, ignore_index=True)


def panel_results(
    paired: pd.DataFrame,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []
    scale_rows: list[dict[str, Any]] = []
    for panel_id, panel in PANELS.items():
        session_sets = [
            set(paired.loc[paired["compound"].eq(compound), "held_session"].unique())
            for compound in panel
        ]
        sessions = sorted(set.intersection(*session_sets))
        subset = paired.loc[
            paired["compound"].isin(panel) & paired["held_session"].isin(sessions)
        ].copy()
        for normalization in ("baseline_mse", "target_variance"):
            scales: dict[str, float] = {}
            for compound in panel:
                group = subset.loc[subset["compound"].eq(compound)]
                baseline_mse = float(group["baseline_squared_error"].mean())
                target_variance = float(group["y_true"].var(ddof=0))
                scale = baseline_mse if normalization == "baseline_mse" else target_variance
                if not math.isfinite(scale) or scale <= 0:
                    raise ValueError(f"Invalid {normalization} scale for {compound}: {scale}")
                scales[compound] = scale
                scale_rows.append(
                    {
                        "panel_id": panel_id,
                        "normalization": normalization,
                        "compound": compound,
                        "scale": scale,
                        "baseline_mse": baseline_mse,
                        "target_variance": target_variance,
                    }
                )
            session_analyte = (
                subset.groupby(["held_session", "compound"], observed=True)[
                    "squared_error_reduction"
                ]
                .mean()
                .rename("mean_squared_error_reduction")
                .reset_index()
            )
            session_analyte["normalization_scale"] = session_analyte["compound"].map(scales)
            session_analyte["normalized_error_reduction"] = (
                session_analyte["mean_squared_error_reduction"]
                / session_analyte["normalization_scale"]
            )
            pivot = session_analyte.pivot(
                index="held_session",
                columns="compound",
                values="normalized_error_reduction",
            ).reindex(columns=list(panel))
            if pivot.isna().any().any():
                raise ValueError("Incomplete LOSO session-by-endpoint panel")
            scores = pivot.mean(axis=1).rename("panel_score").reset_index()
            scores.insert(0, "panel_id", panel_id)
            scores.insert(1, "normalization", normalization)
            score_rows.append(scores)
            values = scores["panel_score"].to_numpy(dtype=float)
            low, high = bootstrap_interval(
                values,
                bootstraps,
                final.stable_seed(seed, panel_id, normalization, "panel_bootstrap"),
            )
            result_rows.append(
                {
                    "panel_id": panel_id,
                    "included_compounds": ";".join(panel),
                    "n_endpoints": len(panel),
                    "normalization": normalization,
                    "n_sessions": len(sessions),
                    "session_equal_normalized_error_reduction": float(np.mean(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "exact_signflip_p": exact_signflip_pvalue(values),
                    "sign_assignments": 2 ** len(values),
                }
            )
    return (
        pd.DataFrame(result_rows),
        pd.concat(score_rows, ignore_index=True),
        pd.DataFrame(scale_rows),
    )
