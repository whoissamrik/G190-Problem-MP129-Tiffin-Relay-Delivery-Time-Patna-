"""Unit tests for data quality checks, cleaning, and PSI (synthetic in-memory data)."""

import numpy as np
import pandas as pd

from tiffin import data_check as dc
from tiffin.clean import clean_training_data
from tiffin.config import load_params
from tiffin.monitor import compute_psi


def _sample_dirty_df() -> pd.DataFrame:
    rows = [
        {
            "record_id": f"R{i:03d}",
            "date": f"2026-01-{i + 1:02d}",
            "distance_km": 5.0 + i,
            "tiffins": 20 + i,
            "handovers": 2,
            "wait_at_station_min": 600.0 if i % 3 == 0 else 10.0,
            "rider_age": 30 + i,
            "temp_c": 25.0 + i,
            "transport": "scooter",
            "route_zone": "Old City" if i % 3 == 0 else ("Central" if i % 3 == 1 else "Suburb"),
            "delivery_min": 400.0 if i == 0 else 40.0 + i,
        }
        for i in range(9)
    ]
    # Add 2 exact duplicate rows
    rows.append(rows[1].copy())
    rows.append(rows[2].copy())
    return pd.DataFrame(rows)


def test_data_checks_detect_issues():
    df = _sample_dirty_df()
    num_cols = ["distance_km", "tiffins", "handovers", "wait_at_station_min", "rider_age", "temp_c"]

    dup = dc.duplicate_rows(df)
    assert dup["present"] is True
    assert dup["count"] == 2

    mix = dc.unit_mixup(df, num_cols, "route_zone")
    assert mix["present"] is True
    assert mix["column"] == "wait_at_station_min"
    assert mix["group_value"] == "Old City"
    assert mix["factor"] == 60

    typos, mask = dc.target_typos(df, "delivery_min")
    assert typos["present"] is True
    assert typos["count"] == 1
    assert int(mask.sum()) == 1


def test_clean_training_data_fixes_issues():
    params = load_params()
    dirty = _sample_dirty_df()
    cleaned = clean_training_data(dirty, params)

    assert len(cleaned) == 9
    assert cleaned["delivery_min"].max() < 100.0
    old_city_wait = cleaned.loc[cleaned["route_zone"] == "Old City", "wait_at_station_min"]
    assert old_city_wait.max() <= 20.0


def test_compute_psi_detects_shift():
    rng = np.random.default_rng(42)
    base = rng.normal(25.0, 5.0, size=500)
    same = rng.normal(25.0, 5.0, size=500)
    shifted = rng.normal(50.0, 5.0, size=500)

    psi_low = compute_psi(base, same, bins=10)
    psi_high = compute_psi(base, shifted, bins=10)

    assert psi_low < 0.2
    assert psi_high > 1.0
