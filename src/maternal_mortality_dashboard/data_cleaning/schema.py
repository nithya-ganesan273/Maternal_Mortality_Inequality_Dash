from __future__ import annotations

import pandera as pa
from pandera import Check, Column, DataFrameSchema

RAW_INDICATOR_SCHEMA = DataFrameSchema(
    {
        "country_iso3": Column(str, checks=Check.str_matches(r"^[A-Z]{3}$")),
        "country_name": Column(str, nullable=False),
        "year": Column(int, checks=Check.in_range(1960, 2100)),
        "metric": Column(
            str,
            checks=Check.isin(
                [
                    "mmr",
                    "gdp_per_capita",
                    "female_secondary_completion",
                    "female_literacy_rate",
                    "health_expenditure_per_capita",
                    "skilled_birth_attendance",
                    "urban_population_pct",
                ]
            ),
        ),
        "value": Column(float, nullable=True, coerce=True),
        "region": Column(str, nullable=True),
        "income_level": Column(str, nullable=True),
        "lending_type": Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)

CLEAN_PANEL_SCHEMA = DataFrameSchema(
    {
        "country_iso3": Column(str, checks=Check.str_matches(r"^[A-Z]{3}$")),
        "country_name": Column(str, nullable=False),
        "region": Column(str, nullable=True),
        "income_level": Column(str, nullable=True),
        "lending_type": Column(str, nullable=True),
        "year": Column(int, checks=Check.in_range(1960, 2100)),
        "mmr": Column(float, nullable=False, checks=Check.gt(0)),
        "gdp_per_capita": Column(float, nullable=True, checks=Check.ge(0)),
        "female_secondary_completion": Column(float, nullable=True, checks=Check.in_range(0, 100)),
        "female_literacy_rate": Column(float, nullable=True, checks=Check.in_range(0, 100)),
        "health_expenditure_per_capita": Column(float, nullable=True, checks=Check.ge(0)),
        "skilled_birth_attendance": Column(float, nullable=True, checks=Check.in_range(0, 100)),
        "urban_population_pct": Column(float, nullable=True, checks=Check.in_range(0, 100)),
        "mmr_rolling3": Column(float, nullable=False, checks=Check.gt(0)),
        "mmr_change_pct": Column(float, nullable=True),
        "log_mmr": Column(float, nullable=False),
    },
    strict=False,
    coerce=True,
)
