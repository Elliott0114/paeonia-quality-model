"""Training-fold fitting and tuning for visible, HSI and joint models."""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from . import chemical as final


MODEL_IDS = (
    "mean_train",
    "visible15_ridge",
    "hsi_P0",
    "visible15_plus_hsi_P0",
    "domain_plus_visible15",
    "domain_plus_visible15_plus_hsi_residual",
)


TUNED_MODEL_IDS = tuple(model for model in MODEL_IDS if model != "mean_train")


def direct_matrix(model_id: str, visible: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    if model_id == "visible15_ridge":
        return visible
    if model_id == "hsi_P0":
        return spectra
    if model_id == "visible15_plus_hsi_P0":
        return np.column_stack([visible, spectra])
    raise ValueError(f"No direct matrix for {model_id}")


def fit_predict(
    model_id: str,
    table: pd.DataFrame,
    visible: np.ndarray,
    spectra: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    alpha: float | None,
) -> np.ndarray:
    observed_mask = np.isfinite(y[train_idx])
    observed_train = train_idx[observed_mask]
    if len(observed_train) < 10:
        raise ValueError("Fewer than 10 strict-range training targets")
    if model_id == "mean_train":
        return np.full(len(predict_idx), float(np.mean(y[observed_train])))
    if alpha is None:
        raise ValueError(f"{model_id} requires alpha")

    if model_id == "hsi_P0":
        return final.fit_predict_model(
            "hsi_P0",
            table,
            spectra,
            y,
            train_idx,
            predict_idx,
            final.DOMAIN_COLUMNS,
            alpha,
        )

    if model_id in {"visible15_ridge", "visible15_plus_hsi_P0"}:
        matrix = direct_matrix(model_id, visible, spectra)
        scaler = StandardScaler().fit(matrix[observed_train])
        model = Ridge(alpha=float(alpha)).fit(
            scaler.transform(matrix[observed_train]), y[observed_train]
        )
        return np.asarray(model.predict(scaler.transform(matrix[predict_idx]))).reshape(-1)

    train_domain, predict_domain, _ = final.encode_domains(
        table,
        train_idx,
        predict_idx,
        final.DOMAIN_COLUMNS,
    )
    visible_scaler = StandardScaler().fit(visible[train_idx])
    nuisance_train = np.column_stack(
        [train_domain, visible_scaler.transform(visible[train_idx])]
    )
    nuisance_predict = np.column_stack(
        [predict_domain, visible_scaler.transform(visible[predict_idx])]
    )
    nuisance_observed = nuisance_train[observed_mask]
    target_model = Ridge(alpha=float(alpha)).fit(nuisance_observed, y[observed_train])
    baseline_prediction = np.asarray(target_model.predict(nuisance_predict)).reshape(-1)
    if model_id == "domain_plus_visible15":
        return baseline_prediction
    if model_id != "domain_plus_visible15_plus_hsi_residual":
        raise ValueError(f"Unknown model: {model_id}")

    spectral_nuisance = LinearRegression().fit(nuisance_train, spectra[train_idx])
    observed_residual_spectra = spectra[observed_train] - spectral_nuisance.predict(
        nuisance_observed
    )
    predict_residual_spectra = spectra[predict_idx] - spectral_nuisance.predict(
        nuisance_predict
    )
    observed_residual_target = y[observed_train] - target_model.predict(nuisance_observed)
    residual_scaler = StandardScaler().fit(observed_residual_spectra)
    residual_model = Ridge(alpha=float(alpha)).fit(
        residual_scaler.transform(observed_residual_spectra),
        observed_residual_target,
    )
    residual_prediction = residual_model.predict(
        residual_scaler.transform(predict_residual_spectra)
    )
    return baseline_prediction + np.asarray(residual_prediction).reshape(-1)


def select_alpha(
    model_id: str,
    table: pd.DataFrame,
    visible: np.ndarray,
    spectra: np.ndarray,
    y: np.ndarray,
    inner_splits: tuple[tuple[np.ndarray, np.ndarray], ...],
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
            predictions.append(
                fit_predict(
                    model_id,
                    table,
                    visible,
                    spectra,
                    y,
                    train_idx,
                    observed_valid,
                    alpha,
                )
            )
            truths.append(y[observed_valid])
        if truths:
            truth = np.concatenate(truths)
            prediction = np.concatenate(predictions)
            rmse = float(math.sqrt(mean_squared_error(truth, prediction)))
            n_valid = int(len(truth))
        else:
            rmse = math.inf
            n_valid = 0
        rows.append({"alpha": alpha, "inner_rmse": rmse, "n_valid": n_valid})
    valid = [row for row in rows if math.isfinite(float(row["inner_rmse"]))]
    if not valid:
        raise RuntimeError(f"No valid alpha for {model_id}")
    selected = min(valid, key=lambda row: (float(row["inner_rmse"]), -float(row["alpha"])))
    scores = pd.DataFrame(rows)
    scores["selected"] = scores["alpha"].eq(float(selected["alpha"]))
    return float(selected["alpha"]), scores
