from __future__ import annotations

import pandas as pd

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
