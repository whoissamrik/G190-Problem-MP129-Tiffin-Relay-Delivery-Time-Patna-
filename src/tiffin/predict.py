from __future__ import annotations
"""Generate holdout predictions using the @champion model."""

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from tiffin.config import load_params


def predict_holdout(params: Dict[str, Any] | None = None) -> Path:
    """Predict delivery_min for data/raw/holdout_features.csv and save record_id,prediction CSV."""
    cfg = params if params is not None else load_params()
    holdout_df = pd.read_csv(cfg["paths"]["holdout_features"])
    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    id_col = cfg["features"]["id_col"]

    model = joblib.load(cfg["paths"]["model_path"])
    preds = model.predict(holdout_df[feature_cols])

    out_df = pd.DataFrame(
        {
            "record_id": holdout_df[id_col],
            "prediction": [round(float(p), 4) for p in preds],
        }
    )
    out_path = Path(cfg["paths"]["predictions_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved {len(out_df)} holdout predictions to {out_path}")
    return out_path


def main() -> None:
    predict_holdout()


if __name__ == "__main__":
    main()
