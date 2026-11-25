from __future__ import annotations
"""CLI runner to execute all seven data checks on MP129 raw files."""

import pandas as pd

from tiffin import data_check as dc
from tiffin.config import load_params


def main() -> None:
    params = load_params()
    train = pd.read_csv(params["paths"]["raw_train"])
    later = pd.read_csv(params["paths"]["raw_later"])
    holdout = pd.read_csv(params["paths"]["holdout_features"])

    results = dc.run_all_checks(
        train=train,
        later=later,
        holdout=holdout,
        num_cols=params["features"]["numeric"],
        cat_cols=params["features"]["categorical"],
        target=params["features"]["target_col"],
        group_col=params["features"]["group_col"],
        min_run=params["cleaning"]["stuck_sensor_min_run"],
        gap_ratio=params["cleaning"]["target_typo_gap_ratio"],
    )
    for name, res in results.items():
        print(f"{name:18s}: {res}")


if __name__ == "__main__":
    main()
