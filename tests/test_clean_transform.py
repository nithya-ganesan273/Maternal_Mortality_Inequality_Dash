from __future__ import annotations

import pandas as pd

from maternal_mortality_dashboard.data_cleaning.transform import clean_indicator_panel


def test_clean_indicator_panel_pivots_and_engineers_features() -> None:
    raw = pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "AAA", "AAA", "AAA", "AAA"],
            "country_name": ["Aland"] * 6,
            "year": [2020, 2020, 2020, 2021, 2021, 2021],
            "metric": [
                "mmr",
                "gdp_per_capita",
                "female_secondary_completion",
                "mmr",
                "gdp_per_capita",
                "female_secondary_completion",
            ],
            "value": [18.0, 40000.0, 93.2, 16.0, 42000.0, 94.3],
            "region": ["Europe"] * 6,
            "income_level": ["High income"] * 6,
            "lending_type": ["IBRD"] * 6,
        }
    )

    output = clean_indicator_panel(raw)
    assert {
        "mmr",
        "gdp_per_capita",
        "female_secondary_completion",
        "female_literacy_rate",
        "health_expenditure_per_capita",
        "skilled_birth_attendance",
        "urban_population_pct",
        "mmr_rolling3",
        "log_mmr",
    }.issubset(output.columns)
    assert output["country_iso3"].nunique() == 1
    assert output["year"].tolist() == [2020, 2021]
