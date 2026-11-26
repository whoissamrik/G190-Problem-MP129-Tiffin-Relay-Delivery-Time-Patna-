from __future__ import annotations
"""End-to-end MLOps pipeline runner for MP129.

Runs:
1. Seven data quality checks on data/raw/train.csv
2. Data cleaning -> data/processed/train_clean.csv
3. v1 model training (4 candidates) with MLflow tracking -> sqlite:///mlflow.db
4. PSI drift, per-group MAE, v2 retraining, and champion promotion
5. Holdout predictions -> predictions/holdout_predictions.csv
6. Populates reports/findings.yaml with every verified value
"""

from pathlib import Path

import pandas as pd
import yaml

from tiffin import data_check as dc
from tiffin.config import load_params
from tiffin.monitor import monitor_and_retrain
from tiffin.predict import predict_holdout
from tiffin.train import train_v1


def run_pipeline() -> None:
    cfg = load_params()

    # 1. Data quality checks on raw files
    raw_train = pd.read_csv(cfg["paths"]["raw_train"])
    raw_later = pd.read_csv(cfg["paths"]["raw_later"])
    holdout = pd.read_csv(cfg["paths"]["holdout_features"])

    data_issues = dc.run_all_checks(
        train=raw_train,
        later=raw_later,
        holdout=holdout,
        num_cols=cfg["features"]["numeric"],
        cat_cols=cfg["features"]["categorical"],
        target=cfg["features"]["target_col"],
        group_col=cfg["features"]["group_col"],
        min_run=int(cfg["cleaning"]["stuck_sensor_min_run"]),
        gap_ratio=float(cfg["cleaning"]["target_typo_gap_ratio"]),
    )

    # 2 & 3. Clean + train v1 + register v1
    v1_res = train_v1(cfg)

    # 4. Monitor drift + retrain v2 + promotion decision
    mon_res = monitor_and_retrain(
        v1_model=v1_res["v1_model"],
        clean_df=v1_res["clean_df"],
        v1_version=v1_res["v1_version"],
        params=cfg,
    )

    # 5. Predict holdout with @champion
    predict_holdout(cfg)

    # 6. Write reports/findings.yaml keeping every required key
    findings = {
        "group_id": cfg["project"]["group_id"],
        "problem_id": cfg["project"]["problem_id"],
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": v1_res["experiment_name"],
            "best_run_id": v1_res["best_run_id"],
            "registered_model_name": v1_res["registered_model_name"],
            "champion_version": mon_res["champion_version"],
        },
        "monitoring": {
            "psi": mon_res["psi"],
            "most_drifted_feature": mon_res["most_drifted_feature"],
            "mae_by_group": mon_res["mae_by_group"],
            "worst_served_group": mon_res["worst_served_group"],
            "mae_v1_on_check": mon_res["mae_v1_on_check"],
            "mae_v2_on_check": mon_res["mae_v2_on_check"],
            "decision": mon_res["decision"],
        },
    }

    findings_path = Path(cfg["paths"]["findings_path"])
    findings_path.parent.mkdir(parents=True, exist_ok=True)
    with open(findings_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(findings, fh, sort_keys=False)

    print(f"Pipeline complete. Updated {findings_path}")


if __name__ == "__main__":
    run_pipeline()
