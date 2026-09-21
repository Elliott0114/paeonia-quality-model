"""Retained-range eligibility, grouped ridge models and batch-level inference."""
from __future__ import annotations

import hashlib
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CohortData:
    """Aligned cohort inputs; targets in mg/g and spectra in vendor-signal units.

    table has a RangeIndex and the columns described in docs/DATA_DICTIONARY.md.
    spectra maps P0–P3 to sample-by-wavelength arrays; compounds uses HPLC names.
    """
    table: pd.DataFrame
    spectra: dict[str, np.ndarray]
    compounds: tuple[str, ...]


CATECHIN = "儿茶素"


PENTAGALLOYLGLUCOSE = "没食子酰葡萄糖"


DOMAIN_COLUMNS = ["group_index", "session_code", "hplc_calibration_group"]


CONTRAST_DOMAIN_COLUMNS = ["position_pair", "session_code", "calibration_pair"]


ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


SPLIT_SEEDS = tuple(range(20260711, 20260721))


PAIR_DEFINITIONS = ((1, 2), (1, 3), (2, 3))


MODEL_IDS = ("mean_train", "domain_only", "hsi_P0", "domain_plus_hsi_orthogonal")


@dataclass(frozen=True)
class CVSplit:
    repeat: int
    fold: int
    split_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    inner_splits: tuple[tuple[np.ndarray, np.ndarray], ...]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def calibration_ranges(
    points_path: Path,
    models_path: Path,
) -> pd.DataFrame:
    points = pd.read_csv(points_path)
    models = pd.read_csv(models_path)
    rows: list[dict[str, Any]] = []
    for _, model in models.iterrows():
        group_id = int(model["group_id"])
        compound = str(model["成分"])
        retained_levels = {
            token.strip() for token in str(model["保留level"]).split(",") if token.strip()
        }
        curve = points.loc[points["group_id"].eq(group_id) & points["compound"].eq(compound)].copy()
        retained = curve.loc[curve["level"].isin(retained_levels)]
        if curve.empty or retained.empty:
            raise ValueError(f"Missing calibration points for group={group_id}, {compound}")
        rows.append(
            {
                "hplc_calibration_group": group_id,
                "compound": compound,
                "retained_level_count": int(len(retained)),
                "retained_min_mg_mL": float(retained["concentration_mg_mL"].min()),
                "retained_max_mg_mL": float(retained["concentration_mg_mL"].max()),
                "full_min_mg_mL": float(curve["concentration_mg_mL"].min()),
                "full_max_mg_mL": float(curve["concentration_mg_mL"].max()),
                "reported_r2": float(model["R²"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["compound", "hplc_calibration_group"])


def build_reference_qualification(
    data: CohortData,
    ranges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    range_lookup = {
        (int(row.hplc_calibration_group), str(row.compound)): (
            float(row.retained_min_mg_mL),
            float(row.retained_max_mg_mL),
            float(row.full_min_mg_mL),
            float(row.full_max_mg_mL),
        )
        for row in ranges.itertuples(index=False)
    }
    detail_rows: list[dict[str, Any]] = []
    targets: dict[str, np.ndarray] = {}
    for compound in data.compounds:
        target = data.table[compound].to_numpy(dtype=float)
        strict_target = np.full(len(target), np.nan)
        for index, row in data.table.iterrows():
            group_id = int(row["hplc_calibration_group"])
            lower, upper, full_lower, full_upper = range_lookup[(group_id, compound)]
            value = float(row[compound]) if pd.notna(row[compound]) else math.nan
            concentration = value / 50.0 if math.isfinite(value) else math.nan
            strict = math.isfinite(concentration) and lower <= concentration <= upper
            full = math.isfinite(concentration) and full_lower <= concentration <= full_upper
            if strict:
                strict_target[index] = value
            detail_rows.append(
                {
                    "canonical_id": row["canonical_id"],
                    "sample_number": int(row["sample_number"]),
                    "group_index": int(row["group_index"]),
                    "session_code": row["session_code"],
                    "hplc_calibration_group": group_id,
                    "compound": compound,
                    "observed_mg_g": value,
                    "sample_concentration_mg_mL": concentration,
                    "retained_min_mg_mL": lower,
                    "retained_max_mg_mL": upper,
                    "full_min_mg_mL": full_lower,
                    "full_max_mg_mL": full_upper,
                    "strict_retained_range_eligible": strict,
                    "full_range_eligible": full,
                }
            )
        targets[compound] = strict_target
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby("compound", observed=True)
        .agg(
            n_total=("canonical_id", "size"),
            n_observed=("observed_mg_g", "count"),
            n_strict_retained_range=("strict_retained_range_eligible", "sum"),
            n_full_range=("full_range_eligible", "sum"),
        )
        .reset_index()
    )
    strict_batches = (
        detail.loc[detail["strict_retained_range_eligible"]]
        .groupby("compound", observed=True)["sample_number"]
        .nunique()
        .rename("n_strict_batches")
    )
    complete_triplets = (
        detail.loc[detail["strict_retained_range_eligible"]]
        .groupby(["compound", "sample_number"], observed=True)
        .size()
        .eq(3)
        .groupby(level=0)
        .sum()
        .rename("n_complete_strict_triplets")
    )
    summary = summary.merge(strict_batches, on="compound", how="left").merge(
        complete_triplets, on="compound", how="left"
    )
    summary[["n_strict_batches", "n_complete_strict_triplets"]] = (
        summary[["n_strict_batches", "n_complete_strict_triplets"]].fillna(0).astype(int)
    )
    summary["quantitative_modeling_status"] = np.where(
        summary["compound"].eq(PENTAGALLOYLGLUCOSE),
        "EXCLUDED_NO_RETAINED_RANGE_OBSERVATIONS",
        "ELIGIBLE_SEVEN_ENDPOINT_FAMILY",
    )
    return detail, summary, targets


def build_repeated_splits(
    table: pd.DataFrame,
    *,
    repeats: int,
    outer_folds: int,
    inner_folds: int,
    seeds: Iterable[int],
) -> list[CVSplit]:
    groups = table["sample_number"].to_numpy()
    output: list[CVSplit] = []
    for repeat, seed in enumerate(list(seeds)[:repeats], start=1):
        outer = GroupKFold(n_splits=outer_folds, shuffle=True, random_state=int(seed))
        for fold, (train_idx, test_idx) in enumerate(outer.split(table, groups=groups), start=1):
            train_groups = groups[train_idx]
            inner = GroupKFold(
                n_splits=min(inner_folds, len(np.unique(train_groups))),
                shuffle=True,
                random_state=stable_seed(seed, fold, "inner"),
            )
            inner_pairs: list[tuple[np.ndarray, np.ndarray]] = []
            for inner_train_local, inner_valid_local in inner.split(train_idx, groups=train_groups):
                inner_pairs.append((train_idx[inner_train_local], train_idx[inner_valid_local]))
            output.append(
                CVSplit(
                    repeat=repeat,
                    fold=fold,
                    split_id=f"R{repeat:02d}F{fold:02d}",
                    train_idx=np.asarray(train_idx, dtype=int),
                    test_idx=np.asarray(test_idx, dtype=int),
                    inner_splits=tuple(inner_pairs),
                )
            )
    return output


def encode_domains(
    table: pd.DataFrame,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder]:
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    train = table.iloc[train_idx][columns].astype(str)
    predict = table.iloc[predict_idx][columns].astype(str)
    encoded_train = np.asarray(encoder.fit_transform(train), dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Found unknown categories.*encoded as all zeros",
            category=UserWarning,
        )
        encoded_predict = np.asarray(encoder.transform(predict), dtype=float)
    return encoded_train, encoded_predict, encoder


def fit_predict_model(
    model_id: str,
    table: pd.DataFrame,
    spectra: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    domain_columns: list[str],
    alpha: float | None = None,
) -> np.ndarray:
    observed_train = train_idx[np.isfinite(y[train_idx])]
    if len(observed_train) < 10:
        raise ValueError("Fewer than 10 observed training targets")
    if model_id == "mean_train":
        return np.full(len(predict_idx), float(np.mean(y[observed_train])))

    if model_id == "hsi_P0":
        if alpha is None:
            raise ValueError("hsi_P0 requires alpha")
        scaler = StandardScaler().fit(spectra[observed_train])
        model = Ridge(alpha=float(alpha)).fit(
            scaler.transform(spectra[observed_train]), y[observed_train]
        )
        return np.asarray(model.predict(scaler.transform(spectra[predict_idx]))).reshape(-1)

    all_train_domain, predict_domain, encoder = encode_domains(
        table, train_idx, predict_idx, domain_columns
    )
    observed_local = np.flatnonzero(np.isin(train_idx, observed_train))
    observed_domain = all_train_domain[observed_local]
    y_domain = LinearRegression().fit(observed_domain, y[observed_train])
    domain_prediction = np.asarray(y_domain.predict(predict_domain)).reshape(-1)
    if model_id == "domain_only":
        return domain_prediction
    if model_id != "domain_plus_hsi_orthogonal" or alpha is None:
        raise ValueError(f"Unknown model or missing alpha: {model_id}")

    # Frisch-Waugh-Lovell-style adjustment. Domain encodings are fit inside the training
    # partition. The spectral nuisance model uses predictors from the training partition only.
    spectral_domain = LinearRegression().fit(all_train_domain, spectra[train_idx])
    train_residual_spectra = spectra[observed_train] - spectral_domain.predict(observed_domain)
    predict_residual_spectra = spectra[predict_idx] - spectral_domain.predict(predict_domain)
    train_residual_target = y[observed_train] - y_domain.predict(observed_domain)
    scaler = StandardScaler().fit(train_residual_spectra)
    residual_model = Ridge(alpha=float(alpha)).fit(
        scaler.transform(train_residual_spectra), train_residual_target
    )
    residual_prediction = residual_model.predict(scaler.transform(predict_residual_spectra))
    return domain_prediction + np.asarray(residual_prediction).reshape(-1)


def select_alpha(
    model_id: str,
    table: pd.DataFrame,
    spectra: np.ndarray,
    y: np.ndarray,
    inner_splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    domain_columns: list[str],
    alphas: tuple[float, ...],
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        truths: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for train_idx, valid_idx in inner_splits:
            observed_valid = valid_idx[np.isfinite(y[valid_idx])]
            if len(observed_valid) == 0:
                continue
            prediction = fit_predict_model(
                model_id,
                table,
                spectra,
                y,
                train_idx,
                observed_valid,
                domain_columns,
                alpha,
            )
            truths.append(y[observed_valid])
            predictions.append(prediction)
        if not truths:
            rmse = math.inf
            n_valid = 0
        else:
            truth = np.concatenate(truths)
            prediction = np.concatenate(predictions)
            rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
            n_valid = int(len(truth))
        rows.append({"alpha": alpha, "inner_rmse": rmse, "n_valid": n_valid})
    valid = [row for row in rows if math.isfinite(float(row["inner_rmse"]))]
    if not valid:
        raise RuntimeError(f"No valid alpha for {model_id}")
    # Prefer stronger regularization on exact ties.
    selected = min(valid, key=lambda row: (float(row["inner_rmse"]), -float(row["alpha"])))
    scores = pd.DataFrame(rows)
    scores["selected"] = scores["alpha"].eq(float(selected["alpha"]))
    return float(selected["alpha"]), scores


def run_split_target(
    split: CVSplit,
    compound: str,
    table: pd.DataFrame,
    spectra: np.ndarray,
    y: np.ndarray,
    domain_columns: list[str],
    model_ids: tuple[str, ...],
    alphas: tuple[float, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        alpha: float | None = None
        if model_id in {"hsi_P0", "domain_plus_hsi_orthogonal"}:
            alpha, scores = select_alpha(
                model_id,
                table,
                spectra,
                y,
                split.inner_splits,
                domain_columns,
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
        prediction = fit_predict_model(
            model_id,
            table,
            spectra,
            y,
            split.train_idx,
            split.test_idx,
            domain_columns,
            alpha,
        )
        test = table.iloc[split.test_idx]
        for local_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "compound": compound,
                    "split_id": split.split_id,
                    "repeat": split.repeat,
                    "outer_fold": split.fold,
                    "canonical_id": row.get("canonical_id", row.get("contrast_id")),
                    "sample_number": int(row["sample_number"]),
                    "group_index": row.get("group_index", math.nan),
                    "session_code": row["session_code"],
                    "hplc_calibration_group": row.get(
                        "hplc_calibration_group", row.get("calibration_pair", "")
                    ),
                    "model_id": model_id,
                    "selected_alpha": alpha,
                    "observed_target": bool(np.isfinite(y[split.test_idx[local_index]])),
                    "y_true": (
                        float(y[split.test_idx[local_index]])
                        if np.isfinite(y[split.test_idx[local_index]])
                        else math.nan
                    ),
                    "y_pred": float(prediction[local_index]),
                }
            )
    return prediction_rows, selection_rows


def run_endpoint_family(
    table: pd.DataFrame,
    spectra: np.ndarray,
    targets: dict[str, np.ndarray],
    splits: list[CVSplit],
    domain_columns: list[str],
    model_ids: tuple[str, ...],
    alphas: tuple[float, ...],
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tasks = [(split, compound) for compound in targets for split in splits]
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(run_split_target)(
            split,
            compound,
            table,
            spectra,
            targets[compound],
            domain_columns,
            model_ids,
            alphas,
        )
        for split, compound in tasks
    )
    predictions = pd.DataFrame([row for result in results for row in result[0]])
    selections = pd.DataFrame([row for result in results for row in result[1]])
    return predictions, selections


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3 or np.std(y_true[mask]) == 0 or np.std(y_pred[mask]) == 0:
        return math.nan
    return float(stats.spearmanr(y_true[mask], y_pred[mask]).statistic)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    truth = np.asarray(y_true[mask], dtype=float)
    prediction = np.asarray(y_pred[mask], dtype=float)
    if len(truth) == 0:
        return {name: math.nan for name in ("rmse", "mae", "r2", "spearman")}
    return {
        "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
        "mae": float(mean_absolute_error(truth, prediction)),
        "r2": float(r2_score(truth, prediction)) if len(truth) >= 2 else math.nan,
        "spearman": safe_spearman(truth, prediction),
    }


def repeat_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scored = predictions.loc[predictions["observed_target"]].copy()
    for (compound, model_id, repeat), group in scored.groupby(
        ["compound", "model_id", "repeat"], observed=True
    ):
        result = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        rows.append(
            {
                "compound": compound,
                "model_id": model_id,
                "repeat": int(repeat),
                "n_portions_or_contrasts": int(len(group)),
                "n_batches": int(group["sample_number"].nunique()),
                **result,
            }
        )
    return pd.DataFrame(rows)


def ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.loc[predictions["observed_target"]].copy()
    identity = [
        "compound",
        "model_id",
        "canonical_id",
        "sample_number",
        "group_index",
        "session_code",
        "hplc_calibration_group",
    ]
    return (
        scored.groupby(identity, observed=True, dropna=False)
        .agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", "mean"),
            repeat_count=("repeat", "nunique"),
        )
        .reset_index()
    )


def cluster_bootstrap_metric_rows(
    group: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    batches = np.asarray(sorted(group["sample_number"].unique()))
    indices_by_batch = {
        batch: np.flatnonzero(group["sample_number"].to_numpy() == batch) for batch in batches
    }
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in ("rmse", "mae", "r2", "spearman")}
    truth = group["y_true"].to_numpy(dtype=float)
    prediction = group["y_pred"].to_numpy(dtype=float)
    for _ in range(replicates):
        sampled_batches = rng.choice(batches, size=len(batches), replace=True)
        sampled_idx = np.concatenate([indices_by_batch[batch] for batch in sampled_batches])
        result = metrics(truth[sampled_idx], prediction[sampled_idx])
        for metric, value in result.items():
            if math.isfinite(value):
                draws[metric].append(value)
    return {
        metric: (
            float(np.quantile(values, 0.025)) if values else math.nan,
            float(np.quantile(values, 0.975)) if values else math.nan,
        )
        for metric, values in draws.items()
    }


def ensemble_metric_summary(
    ensemble: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (compound, model_id), group in ensemble.groupby(["compound", "model_id"], observed=True):
        result = metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        intervals = cluster_bootstrap_metric_rows(
            group,
            bootstrap_replicates,
            stable_seed(seed, compound, model_id, "metric_bootstrap"),
        )
        for metric, estimate in result.items():
            low, high = intervals[metric]
            rows.append(
                {
                    "compound": compound,
                    "model_id": model_id,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_portions_or_contrasts": int(len(group)),
                    "n_batches": int(group["sample_number"].nunique()),
                    "bootstrap_replicates": bootstrap_replicates,
                }
            )
    return pd.DataFrame(rows)


def signflip_pvalue(values: np.ndarray, replicates: int, seed: int) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    chunk_size = 10000
    while completed < replicates:
        size = min(chunk_size, replicates - completed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)), replace=True)
        null = np.mean(signs * values, axis=1)
        exceed += int(np.sum(null >= observed))
        completed += size
    return float((exceed + 1) / (replicates + 1))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    valid_idx = np.flatnonzero(np.isfinite(values))
    if len(valid_idx) == 0:
        return pd.Series(output, index=p_values.index)
    order = valid_idx[np.argsort(values[valid_idx])]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    output[order] = np.minimum(adjusted, 1.0)
    return pd.Series(output, index=p_values.index)


def paired_increment_tests(
    predictions: pd.DataFrame,
    baseline_model: str,
    augmented_model: str,
    signflips: int,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    subset = predictions.loc[
        predictions["observed_target"]
        & predictions["model_id"].isin([baseline_model, augmented_model])
    ].copy()
    pivot = subset.pivot_table(
        index=["compound", "repeat", "canonical_id", "sample_number"],
        columns="model_id",
        values=["y_true", "y_pred"],
        aggfunc="first",
    )
    rows: list[dict[str, Any]] = []
    for compound, group in pivot.groupby(level="compound", observed=True):
        truth = group[("y_true", baseline_model)].to_numpy(dtype=float)
        baseline = group[("y_pred", baseline_model)].to_numpy(dtype=float)
        augmented = group[("y_pred", augmented_model)].to_numpy(dtype=float)
        delta = (truth - baseline) ** 2 - (truth - augmented) ** 2
        batches = group.index.get_level_values("sample_number").to_numpy(dtype=int)
        batch_delta = pd.Series(delta).groupby(batches).mean()
        baseline_mse = float(np.mean((truth - baseline) ** 2))
        augmented_mse = float(np.mean((truth - augmented) ** 2))
        rng = np.random.default_rng(stable_seed(seed, compound, "increment_bootstrap"))
        draws = [
            float(np.mean(rng.choice(batch_delta.to_numpy(), size=len(batch_delta), replace=True)))
            for _ in range(bootstrap_replicates)
        ]
        rows.append(
            {
                "compound": compound,
                "baseline_model": baseline_model,
                "augmented_model": augmented_model,
                "n_batches": int(len(batch_delta)),
                "n_prediction_rows": int(len(group)),
                "baseline_mse": baseline_mse,
                "augmented_mse": augmented_mse,
                "mean_batch_mse_reduction": float(batch_delta.mean()),
                "relative_mse_reduction": (
                    float((baseline_mse - augmented_mse) / baseline_mse)
                    if baseline_mse > 0
                    else math.nan
                ),
                "reduction_ci_low": float(np.quantile(draws, 0.025)),
                "reduction_ci_high": float(np.quantile(draws, 0.975)),
                "signflip_p": signflip_pvalue(
                    batch_delta.to_numpy(),
                    signflips,
                    stable_seed(seed, compound, "signflip"),
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["signflip_q_bh"] = benjamini_hochberg(result["signflip_p"])
    return result.sort_values("signflip_q_bh").reset_index(drop=True)


def build_catechin_contrasts(
    data: CohortData,
    strict_catechin: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    spectral_rows: list[np.ndarray] = []
    for sample_number, batch in data.table.groupby("sample_number", sort=True):
        index_by_g = {int(row["group_index"]): int(index) for index, row in batch.iterrows()}
        for left_g, right_g in PAIR_DEFINITIONS:
            left_idx = index_by_g[left_g]
            right_idx = index_by_g[right_g]
            if not (
                math.isfinite(strict_catechin[left_idx])
                and math.isfinite(strict_catechin[right_idx])
            ):
                continue
            left = data.table.iloc[left_idx]
            right = data.table.iloc[right_idx]
            if left["session_code"] != right["session_code"]:
                raise ValueError("Within-batch contrast crosses HSI sessions")
            rows.append(
                {
                    "contrast_id": f"SBS-{int(sample_number):02d}-G{left_g}_minus_G{right_g}",
                    "sample_number": int(sample_number),
                    "position_pair": f"G{left_g}-G{right_g}",
                    "session_code": left["session_code"],
                    "calibration_pair": (
                        f"C{int(left['hplc_calibration_group'])}-"
                        f"C{int(right['hplc_calibration_group'])}"
                    ),
                    "y_contrast": float(strict_catechin[left_idx] - strict_catechin[right_idx]),
                }
            )
            spectral_rows.append(data.spectra["P0"][left_idx] - data.spectra["P0"][right_idx])
    table = pd.DataFrame(rows)
    spectra = np.asarray(spectral_rows, dtype=float)
    if len(table) != len(spectra) or table["contrast_id"].duplicated().any():
        raise ValueError("Invalid catechin contrast table")
    return table, spectra


def contrast_residual_association(
    ensemble: pd.DataFrame,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    pivot = ensemble.pivot_table(
        index=["canonical_id", "sample_number"],
        columns="model_id",
        values=["y_true", "y_pred"],
        aggfunc="first",
    )
    truth = pivot[("y_true", "domain_only")].to_numpy(dtype=float)
    baseline = pivot[("y_pred", "domain_only")].to_numpy(dtype=float)
    augmented = pivot[("y_pred", "domain_plus_hsi_orthogonal")].to_numpy(dtype=float)
    residual_truth = truth - baseline
    spectral_increment = augmented - baseline
    rho = safe_spearman(residual_truth, spectral_increment)
    direction = float(np.mean(np.sign(residual_truth) == np.sign(spectral_increment)))
    batches = pivot.index.get_level_values("sample_number").to_numpy(dtype=int)
    unique_batches = np.unique(batches)
    indices = {batch: np.flatnonzero(batches == batch) for batch in unique_batches}
    rng = np.random.default_rng(seed)
    rho_draws: list[float] = []
    direction_draws: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled = rng.choice(unique_batches, size=len(unique_batches), replace=True)
        sample_idx = np.concatenate([indices[batch] for batch in sampled])
        sampled_rho = safe_spearman(residual_truth[sample_idx], spectral_increment[sample_idx])
        if math.isfinite(sampled_rho):
            rho_draws.append(sampled_rho)
        direction_draws.append(
            float(
                np.mean(
                    np.sign(residual_truth[sample_idx]) == np.sign(spectral_increment[sample_idx])
                )
            )
        )
    return pd.DataFrame(
        [
            {
                "n_contrasts": int(len(truth)),
                "n_batches": int(len(unique_batches)),
                "residual_spearman": rho,
                "residual_spearman_ci_low": float(np.quantile(rho_draws, 0.025)),
                "residual_spearman_ci_high": float(np.quantile(rho_draws, 0.975)),
                "residual_direction_accuracy": direction,
                "direction_accuracy_ci_low": float(np.quantile(direction_draws, 0.025)),
                "direction_accuracy_ci_high": float(np.quantile(direction_draws, 0.975)),
            }
        ]
    )
