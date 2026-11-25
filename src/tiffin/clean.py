from __future__ import annotations
"""Data cleaning pipeline for MP129 raw training data."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from tiffin import data_check as dc
from tiffin.config import load_params


def clean_training_data(
    df: pd.DataFrame, params: Dict[str, Any] | None = None
) -> pd.DataFrame:
    """Apply verified cleaning steps for MP129:
    1. Remove exact duplicate rows.
    2. Fix unit mixup in Old City wait_at_station_min (seconds -> minutes, /60).
    3. Fix 10x target typos in delivery_min.
    4. Impute stuck-sensor tiffins values during the detected stuck date window.
    """
    cfg = params if params is not None else load_params()
    clean_cfg = cfg["cleaning"]
    target_col = cfg["features"]["target_col"]
    group_col = cfg["features"]["group_col"]
    date_col = cfg["features"]["date_col"]
    num_cols = cfg["features"]["numeric"]

    cleaned = df.drop_duplicates().copy()

    # 1. Fix unit mixup (Old City wait_at_station_min recorded in seconds)
    mix_col = clean_cfg["unit_mixup_col"]
    mix_group = clean_cfg["unit_mixup_group"]
    mix_factor = float(clean_cfg["unit_mixup_factor"])
    if mix_col in cleaned.columns and group_col in cleaned.columns:
        mask_mix = cleaned[group_col] == mix_group
        if mask_mix.any():
            other_median = cleaned.loc[~mask_mix, mix_col].median()
            group_median = cleaned.loc[mask_mix, mix_col].median()
            if other_median > 0 and (group_median / other_median) > 5:
                cleaned.loc[mask_mix, mix_col] = cleaned.loc[mask_mix, mix_col] / mix_factor

    # 2. Fix target typos (values typed 10x too large)
    if target_col in cleaned.columns and len(cleaned) > 1:
        typos_info, typo_mask = dc.target_typos(
            cleaned, target_col, gap_ratio=float(clean_cfg["target_typo_gap_ratio"])
        )
        if typos_info["present"]:
            cleaned.loc[typo_mask, target_col] = (
                cleaned.loc[typo_mask, target_col] / float(clean_cfg["target_typo_factor"])
            )

    # 3. Fix stuck sensor run by imputing with the median of non-stuck rows
    if date_col in cleaned.columns and "record_id" in cleaned.columns:
        stuck_info = dc.stuck_sensor(
            cleaned, num_cols, min_run=int(clean_cfg["stuck_sensor_min_run"])
        )
        if stuck_info["present"]:
            scol = stuck_info["column"]
            stuck_mask = (
                (cleaned[date_col] >= stuck_info["start_date"])
                & (cleaned[date_col] <= stuck_info["end_date"])
            )
            if (~stuck_mask).any():
                median_val = float(cleaned.loc[~stuck_mask, scol].median())
                cleaned.loc[stuck_mask, scol] = median_val

    return cleaned.reset_index(drop=True)


def main() -> None:
    params = load_params()
    raw_df = pd.read_csv(params["paths"]["raw_train"])
    cleaned_df = clean_training_data(raw_df, params)
    out_path = Path(params["paths"]["cleaned_train"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_path, index=False)
    print(f"Cleaned {len(raw_df)} -> {len(cleaned_df)} rows saved to {out_path}")


if __name__ == "__main__":
    main()
