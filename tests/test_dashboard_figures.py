from __future__ import annotations

import pandas as pd

from maternal_mortality_dashboard.dashboard.figures import (
    choropleth_figure,
    country_trend_figure,
    mmr_vs_gdp_figure,
)


def test_country_trend_falls_back_to_mmr_when_selected_metric_missing() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "AAA"],
            "country_name": ["Aland", "Aland", "Aland"],
            "year": [2020, 2021, 2022],
            "mmr": [80.0, 75.0, 70.0],
            "female_secondary_completion": [None, None, None],
        }
    )
    figure = country_trend_figure(frame, "AAA", "female_secondary_completion")
    assert "showing MMR" in str(figure.layout.title.text)
    assert len(figure.data) > 0


def test_choropleth_returns_empty_figure_when_metric_values_missing() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB"],
            "country_name": ["Aland", "Borland"],
            "region": ["Europe", "Europe"],
            "income_level": ["High income", "High income"],
            "gdp_per_capita": [None, None],
        }
    )
    figure = choropleth_figure(frame, "gdp_per_capita", 2021)
    assert len(figure.data) == 0
    assert "No data available" in str(figure.layout.annotations[0]["text"])


def test_scatter_figure_uses_fallback_size_values() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB"],
            "country_name": ["Aland", "Borland"],
            "income_level": ["High income", "Low income"],
            "gdp_per_capita": [20000.0, 1200.0],
            "mmr": [20.0, 180.0],
            "female_secondary_completion": [None, None],
        }
    )
    figure = mmr_vs_gdp_figure(frame, 2021)
    assert len(figure.data) > 0
