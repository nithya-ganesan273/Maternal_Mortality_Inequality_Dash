from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from maternal_mortality_dashboard.data_cleaning.validation import (
    validate_clean_panel,
    validate_raw_indicator_frame,
)
from maternal_mortality_dashboard.exceptions import DataCleaningError

logger = logging.getLogger(__name__)


def clean_indicator_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-form indicators into a clean country-year modeling panel."""

    try:
        validated = validate_raw_indicator_frame(raw_df)
        deduped = validated.drop_duplicates(
            subset=["country_iso3", "year", "metric"],
            keep="last",
        )

        panel = (
            deduped.pivot_table(
                index=[
                    "country_iso3",
                    "country_name",
                    "region",
                    "income_level",
                    "lending_type",
                    "year",
                ],
                columns="metric",
                values="value",
                aggfunc="mean",
            )
            .reset_index()
            .rename_axis(columns=None)
        )

        expected_columns = [
            "mmr",
            "gdp_per_capita",
            "female_secondary_completion",
            "female_literacy_rate",
            "health_expenditure_per_capita",
            "skilled_birth_attendance",
            "urban_population_pct",
        ]
        for column in expected_columns:
            if column not in panel.columns:
                panel[column] = np.nan

        panel = panel.sort_values(["country_iso3", "year"]).reset_index(drop=True)
        numeric_columns = [
            "mmr",
            "gdp_per_capita",
            "female_secondary_completion",
            "female_literacy_rate",
            "health_expenditure_per_capita",
            "skilled_birth_attendance",
            "urban_population_pct",
        ]
        for column in numeric_columns:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

        panel = panel.loc[panel["mmr"].notna() & (panel["mmr"] > 0)].copy()

        interpolation_columns = [
            "gdp_per_capita",
            "female_secondary_completion",
            "female_literacy_rate",
            "health_expenditure_per_capita",
            "urban_population_pct",
        ]
        for column in interpolation_columns:
            panel[column] = panel.groupby("country_iso3", group_keys=False)[column].apply(
                lambda series: series.interpolate(limit_direction="both")
            )

        panel["skilled_birth_attendance"] = panel.groupby("country_iso3", group_keys=False)[
            "skilled_birth_attendance"
        ].apply(lambda series: series.interpolate(limit_area="inside"))

        panel["mmr_rolling3"] = panel.groupby("country_iso3", group_keys=False)["mmr"].apply(
            lambda series: series.rolling(window=3, min_periods=1).mean()
        )
        panel["mmr_change_pct"] = panel.groupby("country_iso3")["mmr"].pct_change() * 100.0
        panel["log_mmr"] = np.log(panel["mmr"])

        logger.info("Cleaned modeling panel built with %s rows", len(panel))
        return validate_clean_panel(panel)
    except Exception as exc:
        raise DataCleaningError("Failed to clean indicator panel") from exc
