from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from maternal_mortality_dashboard.exceptions import PipelineIOError

logger = logging.getLogger(__name__)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
        logger.info("Wrote %s rows to %s", len(df), path)
    except Exception as exc:
        raise PipelineIOError(f"Failed to write parquet artifact: {path}") from exc


def read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise PipelineIOError(f"Parquet artifact not found: {path}") from exc
    except Exception as exc:
        raise PipelineIOError(f"Failed to read parquet artifact: {path}") from exc


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
        logger.info("Wrote JSON artifact to %s", path)
    except Exception as exc:
        raise PipelineIOError(f"Failed to write JSON artifact: {path}") from exc
