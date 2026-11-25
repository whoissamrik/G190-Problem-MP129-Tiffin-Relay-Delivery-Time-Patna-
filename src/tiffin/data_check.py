from __future__ import annotations
"""Diagnostic checks for the seven data quality issues in MP129."""

import numpy as np
import pandas as pd

STANDARD_UNIT_FACTORS = (10, 12, 60, 100, 1000)


def _snap_unit_factor(ratio: float) -> float:
    """Snap an empirical group mean ratio to the nearest standard unit conversion factor."""
    if ratio < 1.0:
        inv = 1.0 / ratio
        best = min(STANDARD_UNIT_FACTORS, key=lambda f: abs(inv - f))
        return round(1.0 / best, 6)
    return int(min(STANDARD_UNIT_FACTORS, key=lambda f: abs(ratio - f)))


def duplicate_rows(df: pd.DataFrame) -> dict:
    """Number of EXTRA copies: rows you would delete."""
    n = int(df.duplicated().sum())
    return {"present": n > 0, "count": n if n > 0 else None}


def unit_mixup(df: pd.DataFrame, num_cols: list[str], group_col: str) -> dict:
    """Find a column where one group's average is wildly different from the others."""
    for col in num_cols:
        means = df.groupby(group_col)[col].mean()
        typical = means.median()
        if typical == 0:
            continue
        for group, m in means.items():
            ratio = float(m / typical)
            if ratio > 5 or ratio < 1 / 5:
                return {
                    "present": True,
                    "column": col,
                    "group_value": str(group),
                    "factor": _snap_unit_factor(ratio),
                }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def leaking_column(
    train: pd.DataFrame, holdout: pd.DataFrame, target: str, num_cols: list[str]
) -> dict:
    """A column missing from holdout, or almost perfectly correlated with the target."""
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    corr = train[num_cols + [target]].corr()[target].drop(target).abs()
    suspicious = list(corr[corr > 0.95].index)
    cols = sorted(set(missing + suspicious))
    return {"present": len(cols) > 0, "column": cols[0] if cols else None}


def sentinel_value(
    df: pd.DataFrame,
    num_cols: list[str],
    candidates: tuple[int, ...] = (-1, -9, -99, -999, -9999, 999, 9999),
) -> dict:
    """A special number typed instead of a blank, repeated several times."""
    for col in num_cols:
        counts = df[col].value_counts()
        for v in candidates:
            if v in counts.index and counts[v] >= 3:
                return {
                    "present": True,
                    "column": col,
                    "value": int(v),
                    "count": int(counts[v]),
                }
    return {"present": False, "column": None, "value": None, "count": None}


def stuck_sensor(df: pd.DataFrame, num_cols: list[str], min_run: int = 10) -> dict:
    """A column that repeats one value over a long run of consecutive rows (sorted by date)."""
    d = df.sort_values(["date", "record_id"]).reset_index(drop=True)
    best = None
    for col in num_cols:
        run_id = (d[col] != d[col].shift()).cumsum()
        runs = d.groupby(run_id).agg(
            n=(col, "size"), start=("date", "first"), end=("date", "last")
        )
        top = runs.sort_values("n", ascending=False).iloc[0]
        if top["n"] >= min_run and (best is None or top["n"] > best[1]):
            best = (col, int(top["n"]), str(top["start"]), str(top["end"]))
    if best:
        return {
            "present": True,
            "column": best[0],
            "start_date": best[2],
            "end_date": best[3],
        }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def target_typos(df: pd.DataFrame, target: str, gap_ratio: float = 2.5) -> tuple[dict, pd.Series]:
    """Find the biggest jump between neighbouring sorted target values; everything above it is a typo."""
    y = np.sort(df[target].values)
    ratios = y[1:] / np.where(y[:-1] == 0, 1e-9, y[:-1])
    i = int(np.argmax(ratios))
    if ratios[i] < gap_ratio:
        return {"present": False, "count": None}, pd.Series(False, index=df.index)
    mask = df[target] > y[i]
    return {"present": True, "count": int(mask.sum())}, mask


def spelling_variants(
    train: pd.DataFrame, later: pd.DataFrame, text_cols: list[str]
) -> dict:
    """Compare the category spellings in train against the clean later.csv."""
    for col in text_cols:
        if set(train[col].unique()) != set(later[col].unique()):
            return {
                "present": True,
                "column": col,
                "true_levels": int(later[col].nunique()),
            }
    return {"present": False, "column": None, "true_levels": None}


def run_all_checks(
    train: pd.DataFrame,
    later: pd.DataFrame,
    holdout: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target: str = "delivery_min",
    group_col: str = "route_zone",
    min_run: int = 10,
    gap_ratio: float = 2.5,
) -> dict:
    """Run all seven diagnostic checks and return a findings.yaml-compatible dictionary."""
    typos_info, _ = target_typos(train, target, gap_ratio=gap_ratio)
    return {
        "leaking_column": leaking_column(train, holdout, target, num_cols),
        "unit_mixup": unit_mixup(train, num_cols, group_col),
        "sentinel_value": sentinel_value(train, num_cols),
        "duplicate_rows": duplicate_rows(train),
        "stuck_sensor": stuck_sensor(train, num_cols, min_run=min_run),
        "target_typos": typos_info,
        "spelling_variants": spelling_variants(train, later, cat_cols),
    }
