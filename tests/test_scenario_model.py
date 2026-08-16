from __future__ import annotations

import math

import pandas as pd
import pytest

from maternal_mortality_dashboard.dashboard.scenario_model import predict_adjusted_mmr


def _coefficient_table(include_skilled: bool = True) -> pd.DataFrame:
    terms = [
        {"term": "const", "coefficient": 7.0},
        {"term": "log_gdp_per_capita", "coefficient": -0.3},
        {"term": "female_literacy_rate", "coefficient": -0.01},
        {"term": "health_expenditure_per_capita", "coefficient": -0.0004},
        {"term": "urban_population_pct", "coefficient": -0.005},
    ]
    if include_skilled:
        terms.append({"term": "skilled_birth_attendance", "coefficient": -0.006})
    return pd.DataFrame(terms)


def test_predict_adjusted_mmr_returns_percent_change_relative_to_baseline() -> None:
    row = pd.Series(
        {
            "gdp_per_capita": 12000.0,
            "female_literacy_rate": 70.0,
            "health_expenditure_per_capita": 800.0,
            "skilled_birth_attendance": 60.0,
            "urban_population_pct": 55.0,
        }
    )

    prediction = predict_adjusted_mmr(
        row=row,
        coefficient_table=_coefficient_table(include_skilled=True),
        literacy_value=80.0,
        health_expenditure_value=1200.0,
        skilled_birth_value=80.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=1000.0,
        fallback_skilled_birth=50.0,
    )

    assert prediction.baseline_mmr is not None
    assert prediction.adjusted_mmr is not None
    assert prediction.percent_change is not None
    assert prediction.percent_change < 0


def test_predict_adjusted_mmr_ignores_skilled_slider_when_not_in_model() -> None:
    row = pd.Series(
        {
            "gdp_per_capita": 9000.0,
            "female_literacy_rate": 65.0,
            "health_expenditure_per_capita": 500.0,
            "skilled_birth_attendance": 30.0,
            "urban_population_pct": 48.0,
        }
    )

    prediction_a = predict_adjusted_mmr(
        row=row,
        coefficient_table=_coefficient_table(include_skilled=False),
        literacy_value=65.0,
        health_expenditure_value=500.0,
        skilled_birth_value=30.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=1000.0,
        fallback_skilled_birth=50.0,
    )
    prediction_b = predict_adjusted_mmr(
        row=row,
        coefficient_table=_coefficient_table(include_skilled=False),
        literacy_value=65.0,
        health_expenditure_value=500.0,
        skilled_birth_value=90.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=1000.0,
        fallback_skilled_birth=50.0,
    )

    assert prediction_a.adjusted_mmr == prediction_b.adjusted_mmr
    assert "not included in the fitted model" in prediction_a.note


def test_predict_adjusted_mmr_handles_missing_country_row() -> None:
    prediction = predict_adjusted_mmr(
        row=None,
        coefficient_table=_coefficient_table(include_skilled=True),
        literacy_value=75.0,
        health_expenditure_value=900.0,
        skilled_birth_value=85.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=1000.0,
        fallback_skilled_birth=50.0,
    )

    assert prediction.baseline_mmr is None
    assert prediction.adjusted_mmr is None
    assert prediction.percent_change is None


def test_predict_adjusted_mmr_returns_none_when_gdp_non_positive() -> None:
    row = pd.Series(
        {
            "gdp_per_capita": 0.0,
            "female_literacy_rate": 70.0,
            "health_expenditure_per_capita": 800.0,
            "skilled_birth_attendance": 60.0,
            "urban_population_pct": 55.0,
        }
    )

    prediction = predict_adjusted_mmr(
        row=row,
        coefficient_table=_coefficient_table(include_skilled=True),
        literacy_value=80.0,
        health_expenditure_value=1200.0,
        skilled_birth_value=80.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=1000.0,
        fallback_skilled_birth=50.0,
    )

    assert prediction.adjusted_mmr is None
    assert prediction.percent_change is None


def test_within_country_scenario_anchors_on_observed_mmr() -> None:
    """
    When the row carries an observed MMR, the baseline must be that value.

    Predicting an absolute level from the model intercept instead would answer a
    between-country question ("what would a country like this look like") while
    the sliders ask a within-country one ("what if this country changed").
    """
    row = pd.Series(
        {
            "mmr": 400.0,
            "gdp_per_capita": 1200.0,
            "female_literacy_rate": 50.0,
            "health_expenditure_per_capita": 40.0,
            "skilled_birth_attendance": 60.0,
            "urban_population_pct": 40.0,
        }
    )
    prediction = predict_adjusted_mmr(
        row=row,
        coefficient_table=_coefficient_table(include_skilled=True),
        literacy_value=50.0,
        health_expenditure_value=40.0,
        skilled_birth_value=60.0,
        fallback_literacy=50.0,
        fallback_health_expenditure=40.0,
        fallback_skilled_birth=60.0,
    )

    assert prediction.baseline_mmr == 400.0
    # No slider moved, so the adjusted value must equal the baseline exactly.
    assert prediction.adjusted_mmr == pytest.approx(400.0)
    assert prediction.percent_change == pytest.approx(0.0)


def test_within_country_percent_change_is_baseline_independent() -> None:
    """
    log-linear outcome => the same slider change is the same percentage change.

    Two countries with very different mortality must show an identical percent
    change for an identical intervention.
    """
    def predict(observed: float):
        row = pd.Series(
            {
                "mmr": observed,
                "female_literacy_rate": 50.0,
                "health_expenditure_per_capita": 40.0,
                "skilled_birth_attendance": 60.0,
            }
        )
        return predict_adjusted_mmr(
            row=row,
            coefficient_table=_coefficient_table(include_skilled=True),
            literacy_value=50.0,
            health_expenditure_value=40.0,
            skilled_birth_value=90.0,
            fallback_literacy=50.0,
            fallback_health_expenditure=40.0,
            fallback_skilled_birth=60.0,
        )

    low, high = predict(80.0), predict(800.0)
    assert low.percent_change == pytest.approx(high.percent_change)
    assert low.percent_change < 0
    # exp(-0.006 * 30) - 1
    assert low.percent_change == pytest.approx((math.exp(-0.006 * 30) - 1) * 100)
