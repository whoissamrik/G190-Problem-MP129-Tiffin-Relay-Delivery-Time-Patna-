from __future__ import annotations
"""Model training and MLflow experiment tracking for MP129 (v1)."""

import hashlib
import inspect
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tiffin.clean import clean_training_data
from tiffin.config import load_params


def compute_sha256(path: str | Path) -> str:
    """Compute hex SHA-256 hash of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_pipeline(
    model_type: str,
    model_params: Dict[str, Any],
    num_cols: list[str],
    cat_cols: list[str],
) -> Pipeline:
    """Build a scikit-learn preprocessing + regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )

    if model_type == "ridge":
        estimator = Ridge(**model_params)
    elif model_type == "random_forest":
        estimator = RandomForestRegressor(**model_params)
    elif model_type == "gradient_boosting":
        estimator = GradientBoostingRegressor(**model_params)
    elif model_type == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(**model_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def log_sklearn_model_compat(model: Pipeline, artifact_path: str = "model") -> None:
    """Log sklearn model to MLflow, passing serialization_format='cloudpickle' when supported."""
    sig = inspect.signature(mlflow.sklearn.log_model)
    kwargs: Dict[str, Any] = {}
    if "serialization_format" in sig.parameters:
        kwargs["serialization_format"] = "cloudpickle"
    if "name" in sig.parameters and "artifact_path" not in sig.parameters:
        mlflow.sklearn.log_model(model, name=artifact_path, **kwargs)
    else:
        mlflow.sklearn.log_model(model, artifact_path=artifact_path, **kwargs)


def train_v1(params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Train 3+ candidate models on cleaned train.csv, track in MLflow, register v1."""
    cfg = params if params is not None else load_params()
    raw_train_path = Path(cfg["paths"]["raw_train"])
    cleaned_train_path = Path(cfg["paths"]["cleaned_train"])
    model_out_path = Path(cfg["paths"]["model_path"])

    data_sha256 = compute_sha256(raw_train_path)
    raw_df = pd.read_csv(raw_train_path)
    clean_df = clean_training_data(raw_df, cfg)
    cleaned_train_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(cleaned_train_path, index=False)

    num_cols = cfg["features"]["numeric"]
    cat_cols = cfg["features"]["categorical"]
    feature_cols = num_cols + cat_cols
    target_col = cfg["features"]["target_col"]

    X = clean_df[feature_cols]
    y = clean_df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(cfg["training"]["val_size"]),
        random_state=int(cfg["project"]["seed"]),
    )

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    best_mae = float("inf")
    best_run_id: str | None = None
    best_candidate: Dict[str, Any] | None = None
    best_model: Pipeline | None = None

    for candidate in cfg["training"]["candidates"]:
        cname = candidate["name"]
        mtype = candidate["type"]
        mparams = candidate.get("params", {})

        with mlflow.start_run(run_name=f"v1_{cname}") as run:
            pipe = build_pipeline(mtype, mparams, num_cols, cat_cols)
            pipe.fit(X_train, y_train)

            train_pred = pipe.predict(X_train)
            val_pred = pipe.predict(X_val)
            mae_train = float(mean_absolute_error(y_train, train_pred))
            mae_val = float(mean_absolute_error(y_val, val_pred))

            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_version_stage", "v1_candidate")
            mlflow.log_param("model_type", mtype)
            for k, v in mparams.items():
                mlflow.log_param(k, v)

            mlflow.log_metric("mae_train", mae_train)
            mlflow.log_metric("mae_val", mae_val)

            # Refit on full cleaned train before logging artifact for registration
            full_pipe = build_pipeline(mtype, mparams, num_cols, cat_cols)
            full_pipe.fit(X, y)
            log_sklearn_model_compat(full_pipe, artifact_path="model")

            if mae_val < best_mae:
                best_mae = mae_val
                best_run_id = run.info.run_id
                best_candidate = candidate
                best_model = full_pipe

    assert best_run_id is not None and best_model is not None and best_candidate is not None

    reg_name = cfg["mlflow"]["registered_model_name"]
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=reg_name)

    client = MlflowClient()
    client.set_registered_model_alias(reg_name, "champion", mv.version)

    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_out_path)

    return {
        "data_sha256": data_sha256,
        "experiment_name": cfg["mlflow"]["experiment_name"],
        "best_run_id": best_run_id,
        "registered_model_name": reg_name,
        "v1_version": int(mv.version),
        "best_candidate": best_candidate,
        "best_val_mae": round(best_mae, 4),
        "v1_model": best_model,
        "clean_df": clean_df,
    }


def main() -> None:
    res = train_v1()
    print(
        f"Registered v1 (version {res['v1_version']}) from run {res['best_run_id']} "
        f"with val MAE={res['best_val_mae']}"
    )


if __name__ == "__main__":
    main()
