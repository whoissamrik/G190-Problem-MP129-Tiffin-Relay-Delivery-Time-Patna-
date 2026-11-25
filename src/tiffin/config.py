from __future__ import annotations
"""Configuration loader for MP129."""

from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path | str | None = None) -> Dict[str, Any]:
    """Load pipeline parameters from params.yaml using yaml.safe_load."""
    params_path = Path(path) if path is not None else DEFAULT_PARAMS_PATH
    with open(params_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
