from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from maternal_mortality_dashboard.exceptions import ModelingError
from maternal_mortality_dashboard.modeling.inequality_metrics import (
    _gini,
    build_latest_country_snapshot,
    compute_income_group_summary,
    compute_income_inequality_metrics,
    compute_yearly_inequality,
)


def _sample_clean_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "country_iso3": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "country_name": ["Aland", "Aland", "Borland", "Borland", "Cyprus", "Cyprus"],
            "region": ["Europe", "Europe", "Africa", "Africa", "Asia", "Asia"],
            "income_level": ["High income", "High income", "Low income", "Low income", "Middle income", "Middle income"],
            "lending_type": ["IBRD", "IBRD", "IDA", "IDA", "Blend", "Blend"],
            "year": [2020, 2021, 2020, 2021, 2020, 2021],
            "mmr": [18.0, 16.0, 220.0, 210.0, 90.0, 84.0],
            "gdp_per_capita": [41000.0, 42500.0, 1300.0, 1400.0, 9800.0, 10200.0],
            "female_secondary_completion": [95.0, 95.6, 42.1, 43.8, 71.2, 72.4],
            "mmr_rolling3": [18.0, 17.0, 220.0, 215.0, 90.0, 87.0],
            "mmr_change_pct": [None, -11.11, None, -4.54, None, -6.67],
            "log_mmr": [2.89, 2.77, 5.39, 5.34, 4.50, 4.43],
        }
    )


def test_compute_yearly_inequality_has_expected_columns() -> None:
    frame = _sample_clean_panel()
    output = compute_yearly_inequality(frame)
    assert {
        "year",
        "countries",
        "mmr_median",
        "gini_mmr",
        "p90_p10_ratio",
        "concentration_index_income_group",
        "absolute_low_high_difference",
        "relative_low_high_ratio",
        "between_group_variance",
    }.issubset(output.columns)
    assert output["year"].tolist() == [2020, 2021]
    assert (output["countries"] == 3).all()


def test_compute_yearly_inequality_regression_values() -> None:
    frame = _sample_clean_panel()
    output = compute_yearly_inequality(frame)

    metrics_2020 = output.loc[output["year"] == 2020].iloc[0]
    assert metrics_2020["mmr_mean"] == pytest.approx(109.3333333333, rel=1e-9)
    assert metrics_2020["mmr_median"] == pytest.approx(90.0)
    assert metrics_2020["mmr_p10"] == pytest.approx(32.4, rel=1e-9)
    assert metrics_2020["mmr_p90"] == pytest.approx(194.0, rel=1e-9)
    assert metrics_2020["p90_p10_ratio"] == pytest.approx(5.9876543210, rel=1e-9)
    assert metrics_2020["gini_mmr"] == pytest.approx(0.4105691057, rel=1e-9)
    assert metrics_2020["concentration_index_income_group"] == pytest.approx(-0.4105691057, rel=1e-9)
    assert metrics_2020["absolute_low_high_difference"] == pytest.approx(202.0, rel=1e-9)
    assert metrics_2020["relative_low_high_ratio"] == pytest.approx(12.2222222222, rel=1e-9)
    assert metrics_2020["between_group_variance"] == pytest.approx(6987.5555555556, rel=1e-9)

    metrics_2021 = output.loc[output["year"] == 2021].iloc[0]
    assert metrics_2021["mmr_mean"] == pytest.approx(103.3333333333, rel=1e-9)
    assert metrics_2021["mmr_median"] == pytest.approx(84.0)
    assert metrics_2021["mmr_p10"] == pytest.approx(29.6, rel=1e-9)
    assert metrics_2021["mmr_p90"] == pytest.approx(184.8, rel=1e-9)
    assert metrics_2021["p90_p10_ratio"] == pytest.approx(6.2432432432, rel=1e-9)
    assert metrics_2021["gini_mmr"] == pytest.approx(0.4172043011, rel=1e-9)
    assert metrics_2021["concentration_index_income_group"] == pytest.approx(-0.4172043011, rel=1e-9)
    assert metrics_2021["absolute_low_high_difference"] == pytest.approx(194.0, rel=1e-9)
    assert metrics_2021["relative_low_high_ratio"] == pytest.approx(13.125, rel=1e-9)
    assert metrics_2021["between_group_variance"] == pytest.approx(6459.5555555556, rel=1e-9)


def test_compute_yearly_inequality_all_missing_mmr_returns_nan_metrics() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB"],
            "year": [2022, 2022],
            "mmr": [np.nan, np.nan],
        }
    )

    output = compute_yearly_inequality(frame)
    row = output.iloc[0]
    assert row["countries"] == 0
    assert np.isnan(row["mmr_mean"])
    assert np.isnan(row["mmr_median"])
    assert np.isnan(row["gini_mmr"])
    assert np.isnan(row["p90_p10_ratio"])
    assert np.isnan(row["concentration_index_income_group"])
    assert np.isnan(row["absolute_low_high_difference"])
    assert np.isnan(row["relative_low_high_ratio"])
    assert np.isnan(row["between_group_variance"])


def test_compute_yearly_inequality_missing_required_column_raises_error() -> None:
    frame = pd.DataFrame({"country_iso3": ["AAA"], "year": [2020]})
    with pytest.raises(ModelingError):
        compute_yearly_inequality(frame)


def test_compute_income_inequality_metrics_missing_low_or_high_returns_nan_comparisons() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB", "CCC", "DDD"],
            "year": [2022, 2022, 2022, 2022],
            "income_level": ["Middle income", "Middle income", "Upper middle income", "Upper middle income"],
            "mmr": [100.0, 110.0, 70.0, 80.0],
        }
    )

    output = compute_income_inequality_metrics(frame)
    row = output.iloc[0]
    assert np.isnan(row["absolute_low_high_difference"])
    assert np.isnan(row["relative_low_high_ratio"])
    assert row["between_group_variance"] >= 0


def test_compute_income_inequality_metrics_equal_group_means_yield_zero_inequality() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB", "CCC", "DDD"],
            "year": [2024, 2024, 2024, 2024],
            "income_level": ["Low income", "Lower middle income", "Upper middle income", "High income"],
            "mmr": [50.0, 50.0, 50.0, 50.0],
        }
    )

    output = compute_income_inequality_metrics(frame)
    row = output.iloc[0]
    assert row["concentration_index_income_group"] == pytest.approx(0.0, abs=1e-12)
    assert row["between_group_variance"] == pytest.approx(0.0, abs=1e-12)
    assert row["absolute_low_high_difference"] == pytest.approx(0.0, abs=1e-12)
    assert row["relative_low_high_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_compute_income_inequality_metrics_missing_required_column_raises_error() -> None:
    frame = pd.DataFrame({"country_iso3": ["AAA"], "year": [2020]})
    with pytest.raises(ModelingError):
        compute_income_inequality_metrics(frame)


def test_compute_yearly_inequality_without_income_column_keeps_new_metrics_nan() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB"],
            "year": [2023, 2023],
            "mmr": [60.0, 120.0],
        }
    )
    output = compute_yearly_inequality(frame)
    row = output.iloc[0]
    assert np.isnan(row["concentration_index_income_group"])
    assert np.isnan(row["absolute_low_high_difference"])
    assert np.isnan(row["relative_low_high_ratio"])
    assert np.isnan(row["between_group_variance"])


def test_income_group_summary_contains_each_group() -> None:
    frame = _sample_clean_panel()
    output = compute_income_group_summary(frame)
    filtered = output.loc[output["year"] == 2021]
    assert set(filtered["income_level"]) == {"High income", "Low income", "Middle income"}


def test_income_group_summary_regression_values() -> None:
    frame = _sample_clean_panel()
    output = compute_income_group_summary(frame)
    low_income = output.loc[
        (output["year"] == 2021) & (output["income_level"] == "Low income")
    ].iloc[0]

    assert low_income["countries"] == 1
    assert low_income["mmr_mean"] == pytest.approx(210.0)
    assert low_income["mmr_median"] == pytest.approx(210.0)
    assert low_income["gdp_per_capita_mean"] == pytest.approx(1400.0)
    assert low_income["female_secondary_completion_mean"] == pytest.approx(43.8)


def test_income_group_summary_preserves_missing_income_level_group() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["AAA", "BBB"],
            "year": [2023, 2023],
            "income_level": [None, "Low income"],
            "mmr": [56.0, np.nan],
            "gdp_per_capita": [3200.0, 1200.0],
            "female_secondary_completion": [60.0, 42.0],
        }
    )

    output = compute_income_group_summary(frame)
    assert output["income_level"].isna().sum() == 1

    missing_income_group = output.loc[output["income_level"].isna()].iloc[0]
    assert missing_income_group["countries"] == 1
    assert missing_income_group["mmr_mean"] == pytest.approx(56.0)

    low_income_group = output.loc[output["income_level"] == "Low income"].iloc[0]
    assert low_income_group["countries"] == 1
    assert np.isnan(low_income_group["mmr_mean"])


def test_latest_country_snapshot_returns_latest_year_per_country() -> None:
    frame = _sample_clean_panel()
    output = build_latest_country_snapshot(frame)
    assert output["year"].nunique() == 1
    assert output["year"].iloc[0] == 2021


def test_latest_country_snapshot_selects_latest_year_from_unsorted_input() -> None:
    frame = pd.DataFrame(
        {
            "country_iso3": ["BBB", "AAA", "BBB", "AAA"],
            "country_name": ["Borland", "Aland", "Borland", "Aland"],
            "year": [2019, 2021, 2022, 2020],
            "mmr": [225.0, 15.0, 205.0, 17.0],
        }
    )

    output = build_latest_country_snapshot(frame)
    resolved = dict(zip(output["country_iso3"], output["year"], strict=True))
    assert resolved == {"AAA": 2021, "BBB": 2022}
    assert output["country_name"].tolist() == ["Aland", "Borland"]


def test_gini_regression_and_edge_cases() -> None:
    assert _gini(np.array([18.0, 90.0, 220.0])) == pytest.approx(0.4105691057, rel=1e-9)
    assert _gini(np.array([0.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert np.isnan(_gini(np.array([np.nan, -5.0])))
