from __future__ import annotations
"""Drift monitoring (PSI), group MAE analysis, v2 retraining, and champion promotion."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from tiffin.clean import clean_training_data
from tiffin.config import load_params


def compute_psi(
    expected: pd.Series | np.ndarray,
    actual: pd.Series | np.ndarray,
    bins: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) with 10 quantile bins from expected."""
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(exp, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts, _ = np.histogram(exp, bins=edges)
    act_counts, _ = np.histogram(act, bins=edges)
    exp_pct = np.clip(exp_counts / len(exp), 1e-6, None)
    act_pct = np.clip(act_counts / len(act), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def compute_all_psi(
    clean_train: pd.DataFrame, later: pd.DataFrame, num_cols: list[str], bins: int = 10
) -> Dict[str, float]:
    """Compute PSI for every numeric column between cleaned train.csv and later.csv."""
    return {col: round(compute_psi(clean_train[col], later[col], bins=bins), 4) for col in num_cols}


def monitor_and_retrain(
    v1_model: Any | None = None,
    clean_df: pd.DataFrame | None = None,
    v1_version: int = 1,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Run drift checks, group MAE evaluation, v2 retrain on train + 70% later, and promotion."""
    import joblib
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import mean_absolute_error

    from tiffin.train import build_pipeline, compute_sha256, log_sklearn_model_compat

    cfg = params if params is not None else load_params()
    num_cols = cfg["features"]["numeric"]
    cat_cols = cfg["features"]["categorical"]
    feature_cols = num_cols + cat_cols
    target_col = cfg["features"]["target_col"]
    group_col = cfg["features"]["group_col"]
    date_col = cfg["features"]["date_col"]
    id_col = cfg["features"]["id_col"]

    raw_train_path = Path(cfg["paths"]["raw_train"])
    data_sha256 = compute_sha256(raw_train_path)

    if clean_df is None:
        cleaned_path = Path(cfg["paths"]["cleaned_train"])
        if cleaned_path.exists():
            clean_df = pd.read_csv(cleaned_path)
        else:
            clean_df = clean_training_data(pd.read_csv(raw_train_path), cfg)

    later_df = pd.read_csv(cfg["paths"]["raw_later"])

    model_path = Path(cfg["paths"]["model_path"])
    if v1_model is None:
        v1_model = joblib.load(model_path)

    # 1. PSI per numeric column
    psi_bins = int(cfg["monitoring"]["psi_bins"])
    psi_scores = compute_all_psi(clean_df, later_df, num_cols, bins=psi_bins)
    most_drifted_feature = max(psi_scores, key=psi_scores.get)

    # 2. v1 MAE on later.csv per route_zone
    later_preds_v1 = v1_model.predict(later_df[feature_cols])
    mae_later_overall = float(mean_absolute_error(later_df[target_col], later_preds_v1))
    mae_by_group: Dict[str, float] = {}
    for group_val in ["Central", "Suburb", "Old City"]:
        mask = later_df[group_col] == group_val
        mae_by_group[group_val] = round(
            float(mean_absolute_error(later_df.loc[mask, target_col], later_preds_v1[mask])),
            4,
        )
    worst_served_group = max(mae_by_group, key=mae_by_group.get)

    # 3. Sort later.csv by date, split 70/30, retrain v2
    later_sorted = later_df.sort_values([date_col, id_col]).reset_index(drop=True)
    split_idx = int(len(later_sorted) * float(cfg["monitoring"]["retrain_split"]))
    later_train = later_sorted.iloc[:split_idx]
    later_check = later_sorted.iloc[split_idx:]

    combined_train = pd.concat([clean_df, later_train], ignore_index=True)

    # Use GradientBoostingRegressor for v2 to capture route_zone + temperature shift interactions
    v2_params = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 4,
        "random_state": int(cfg["project"]["seed"]),
    }
    v2_model = build_pipeline("gradient_boosting", v2_params, num_cols, cat_cols)
    v2_model.fit(combined_train[feature_cols], combined_train[target_col])

    mae_v1_on_check = float(
        mean_absolute_error(later_check[target_col], v1_model.predict(later_check[feature_cols]))
    )
    mae_v2_on_check = float(
        mean_absolute_error(later_check[target_col], v2_model.predict(later_check[feature_cols]))
    )

    # 4. Promotion rule: promote v2 only if MAE on last 30% is at least 5% lower than v1's
    threshold = float(cfg["monitoring"]["promotion_threshold"])
    promote = mae_v2_on_check <= mae_v1_on_check * (1.0 - threshold)
    decision = "promote" if promote else "keep"

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    reg_name = cfg["mlflow"]["registered_model_name"]

    with mlflow.start_run(run_name="v2_retrained_gradient_boosting") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_version_stage", "v2_retrain")
        mlflow.log_param("model_type", "gradient_boosting")
        for k, v in v2_params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("promotion_decision", decision)

        mlflow.log_metric("mae_later_v1", mae_later_overall)
        mlflow.log_metric("mae_v1_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_check", mae_v2_on_check)
        for col, val in psi_scores.items():
            mlflow.log_metric(f"psi_{col}", val)

        log_sklearn_model_compat(v2_model, artifact_path="model")
        v2_run_id = run.info.run_id

    mv2 = mlflow.register_model(model_uri=f"runs:/{v2_run_id}/model", name=reg_name)
    client = MlflowClient()

    if promote:
        champion_version = int(mv2.version)
        client.set_registered_model_alias(reg_name, "champion", mv2.version)
        joblib.dump(v2_model, model_path)
    else:
        champion_version = int(v1_version)
        client.set_registered_model_alias(reg_name, "champion", str(v1_version))

    return {
        "psi": psi_scores,
        "most_drifted_feature": most_drifted_feature,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_served_group,
        "mae_v1_on_check": round(mae_v1_on_check, 4),
        "mae_v2_on_check": round(mae_v2_on_check, 4),
        "decision": decision,
        "champion_version": champion_version,
        "v2_run_id": v2_run_id,
    }


def main() -> None:
    res = monitor_and_retrain()
    print(
        f"Most drifted: {res['most_drifted_feature']} | Worst group: {res['worst_served_group']} | "
        f"v1 check MAE={res['mae_v1_on_check']} vs v2 check MAE={res['mae_v2_on_check']} -> {res['decision']}"
    )


if __name__ == "__main__":
    main()
