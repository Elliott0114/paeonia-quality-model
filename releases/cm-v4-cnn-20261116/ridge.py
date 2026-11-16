import math,warnings
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.metrics import mean_squared_error

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
