from __future__ import annotations

import logging

import pandera as pa
import pandas as pd

from maternal_mortality_dashboard.data_cleaning.schema import CLEAN_PANEL_SCHEMA, RAW_INDICATOR_SCHEMA
from maternal_mortality_dashboard.exceptions import DataCleaningError

logger = logging.getLogger(__name__)


def validate_raw_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated = RAW_INDICATOR_SCHEMA.validate(df, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise DataCleaningError("Raw indicator schema validation failed") from exc

    logger.info("Raw indicator frame validation succeeded with %s rows", len(validated))
    return validated


def validate_clean_panel(df: pd.DataFrame) -> pd.DataFrame:
    try:
        validated = CLEAN_PANEL_SCHEMA.validate(df, lazy=True)
    except (pa.errors.SchemaError, pa.errors.SchemaErrors) as exc:
        raise DataCleaningError("Clean panel schema validation failed") from exc

    logger.info("Clean panel validation succeeded with %s rows", len(validated))
    return validated
