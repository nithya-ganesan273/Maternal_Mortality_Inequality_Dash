from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioPrediction:
    baseline_mmr: float | None
    adjusted_mmr: float | None
    percent_change: float | None
    note: str


def _to_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(converted) or np.isinf(converted):
        return None
    return converted


def coefficient_map(coefficient_table: pd.DataFrame) -> dict[str, float]:
    if "term" not in coefficient_table.columns or "coefficient" not in coefficient_table.columns:
        return {}

    table = coefficient_table[["term", "coefficient"]].dropna(subset=["term", "coefficient"])
    table["term"] = table["term"].astype(str)
    return {
        row["term"]: float(row["coefficient"])
        for row in table.to_dict(orient="records")
        if _to_float(row["coefficient"]) is not None
    }


def _predict_mmr_from_coefficients(
    coefficients: dict[str, float],
    *,
    gdp_per_capita: float | None,
    female_literacy_rate: float | None,
    health_expenditure_per_capita: float | None,
    skilled_birth_attendance: float | None,
    urban_population_pct: float | None,
) -> tuple[float | None, str]:
    if not coefficients:
        return None, "Regression coefficients were not available."

    linear_predictor = float(coefficients.get("const", 0.0))
    issues: list[str] = []

    if "log_gdp_per_capita" in coefficients:
        if gdp_per_capita is None or gdp_per_capita <= 0:
            return None, "GDP per capita was non-positive or unavailable for prediction."
        linear_predictor += coefficients["log_gdp_per_capita"] * math.log(gdp_per_capita)

    if "female_literacy_rate" in coefficients:
        if female_literacy_rate is None:
            return None, "Female literacy rate was unavailable for prediction."
        linear_predictor += coefficients["female_literacy_rate"] * female_literacy_rate

    if "health_expenditure_per_capita" in coefficients:
        if health_expenditure_per_capita is None:
            return None, "Health expenditure per capita was unavailable for prediction."
        linear_predictor += coefficients["health_expenditure_per_capita"] * health_expenditure_per_capita

    if "urban_population_pct" in coefficients:
        if urban_population_pct is None:
            return None, "Urban population percentage was unavailable for prediction."
        linear_predictor += coefficients["urban_population_pct"] * urban_population_pct

    if "skilled_birth_attendance" in coefficients:
        if skilled_birth_attendance is None:
            return None, "Skilled birth attendance was unavailable for prediction."
        linear_predictor += coefficients["skilled_birth_attendance"] * skilled_birth_attendance
    else:
        issues.append("Skilled birth attendance was not included in the fitted model.")

    mmr_value = float(math.exp(linear_predictor))
    if not np.isfinite(mmr_value):
        return None, "Model prediction returned a non-finite value."

    note = " ".join(issues) if issues else "Prediction generated using fitted ecological regression coefficients."
    return mmr_value, note


def predict_adjusted_mmr(
    row: pd.Series | None,
    coefficient_table: pd.DataFrame,
    *,
    literacy_value: float | None,
    health_expenditure_value: float | None,
    skilled_birth_value: float | None,
    fallback_literacy: float,
    fallback_health_expenditure: float,
    fallback_skilled_birth: float,
) -> ScenarioPrediction:
    if row is None:
        return ScenarioPrediction(
            baseline_mmr=None,
            adjusted_mmr=None,
            percent_change=None,
            note="No country-year observation was available for scenario prediction.",
        )

    coefficients = coefficient_map(coefficient_table)

    gdp_per_capita = _to_float(row.get("gdp_per_capita"))
    urban_population_pct = _to_float(row.get("urban_population_pct"))
    baseline_literacy = _to_float(row.get("female_literacy_rate")) or fallback_literacy
    baseline_health = _to_float(row.get("health_expenditure_per_capita")) or fallback_health_expenditure
    baseline_skilled = _to_float(row.get("skilled_birth_attendance")) or fallback_skilled_birth

    scenario_literacy = _to_float(literacy_value) or baseline_literacy
    scenario_health = _to_float(health_expenditure_value) or baseline_health
    scenario_skilled = _to_float(skilled_birth_value) or baseline_skilled

    baseline_pred, baseline_note = _predict_mmr_from_coefficients(
        coefficients,
        gdp_per_capita=gdp_per_capita,
        female_literacy_rate=baseline_literacy,
        health_expenditure_per_capita=baseline_health,
        skilled_birth_attendance=baseline_skilled,
        urban_population_pct=urban_population_pct,
    )
    adjusted_pred, adjusted_note = _predict_mmr_from_coefficients(
        coefficients,
        gdp_per_capita=gdp_per_capita,
        female_literacy_rate=scenario_literacy,
        health_expenditure_per_capita=scenario_health,
        skilled_birth_attendance=scenario_skilled,
        urban_population_pct=urban_population_pct,
    )

    if baseline_pred is None or adjusted_pred is None:
        return ScenarioPrediction(
            baseline_mmr=baseline_pred,
            adjusted_mmr=adjusted_pred,
            percent_change=None,
            note=f"{baseline_note} {adjusted_note}".strip(),
        )

    if baseline_pred <= 0:
        percent_change = None
    else:
        percent_change = float(((adjusted_pred - baseline_pred) / baseline_pred) * 100.0)

    note = (
        "Baseline prediction uses observed country-year covariates when available and "
        "median substitution for missing predictors. "
        + adjusted_note
    )
    return ScenarioPrediction(
        baseline_mmr=baseline_pred,
        adjusted_mmr=adjusted_pred,
        percent_change=percent_change,
        note=note,
    )
