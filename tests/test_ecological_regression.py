from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from maternal_mortality_dashboard.exceptions import ModelingError
from maternal_mortality_dashboard.modeling.ecological_regression import (
    OPTIONAL_PREDICTOR,
    run_ecological_regression,
)


def _synthetic_clean_panel(
    n_countries: int = 8,
    start_year: int = 2016,
    end_year: int = 2022,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []
    years = list(range(start_year, end_year + 1))

    for country_index in range(n_countries):
        iso3 = f"C{country_index:02d}"
        country_name = f"Country {country_index:02d}"
        baseline_gdp = rng.uniform(800, 45000)
        baseline_female_lit = rng.uniform(45, 98)
        baseline_health_exp = rng.uniform(40, 2000)
        baseline_urban = rng.uniform(20, 90)
        baseline_sba = rng.uniform(35, 99)

        for year in years:
            year_shift = year - years[0]
            gdp = baseline_gdp * (1.03**year_shift) * rng.uniform(0.95, 1.05)
            female_literacy = np.clip(baseline_female_lit + year_shift * rng.uniform(0.1, 0.5), 5, 100)
            health_expenditure = np.clip(
                baseline_health_exp * (1.04**year_shift) * rng.uniform(0.9, 1.08),
                5,
                None,
            )
            urban_pct = np.clip(baseline_urban + year_shift * rng.uniform(0.05, 0.4), 1, 100)
            sba = np.clip(baseline_sba + year_shift * rng.uniform(0.05, 0.3), 1, 100)

            log_mmr = (
                7.1
                - 0.33 * np.log(gdp)
                - 0.010 * female_literacy
                - 0.00045 * health_expenditure
                - 0.007 * urban_pct
                - 0.006 * sba
                + rng.normal(0.0, 0.045)
            )

            mmr = float(np.exp(log_mmr))
            records.append(
                {
                    "country_iso3": iso3,
                    "country_name": country_name,
                    "year": year,
                    "mmr": mmr,
                    "gdp_per_capita": float(gdp),
                    "female_literacy_rate": float(female_literacy),
                    "health_expenditure_per_capita": float(health_expenditure),
                    "skilled_birth_attendance": float(sba),
                    "urban_population_pct": float(urban_pct),
                }
            )

    return pd.DataFrame.from_records(records)


def test_ecological_regression_generates_expected_artifacts(tmp_path: Path) -> None:
    frame = _synthetic_clean_panel()
    result = run_ecological_regression(
        clean_panel=frame,
        output_dir=tmp_path,
        minimum_observations=40,
        optional_predictor_min_coverage=0.5,
    )

    assert OPTIONAL_PREDICTOR in result.predictors_used
    assert result.n_observations >= 40
    assert result.r_squared > 0.7

    assert result.artifacts.regression_dataset_path.exists()
    assert result.artifacts.coefficients_path.exists()
    assert result.artifacts.model_summary_csv_path.exists()
    assert result.artifacts.model_summary_text_path.exists()
    assert result.artifacts.vif_path.exists()
    assert result.artifacts.residuals_vs_fitted_plot_path.exists()
    assert result.artifacts.qq_plot_path.exists()

    coefficient_table = pd.read_csv(result.artifacts.coefficients_path)
    assert {
        "term",
        "coefficient",
        "robust_std_error",
        "p_value",
        "epidemiologic_interpretation",
    }.issubset(coefficient_table.columns)

    assert "const" in set(coefficient_table["term"])
    for predictor in result.predictors_used:
        assert predictor in set(coefficient_table["term"])

    coeff_map = dict(zip(coefficient_table["term"], coefficient_table["coefficient"]))
    assert coeff_map["log_gdp_per_capita"] < 0
    assert coeff_map["female_literacy_rate"] < 0
    assert coeff_map["health_expenditure_per_capita"] < 0
    assert coeff_map["urban_population_pct"] < 0
    assert coeff_map["skilled_birth_attendance"] < 0

    vif_table = pd.read_csv(result.artifacts.vif_path)
    assert {"variable", "vif", "high_multicollinearity_flag"}.issubset(vif_table.columns)

    summary_text = result.artifacts.model_summary_text_path.read_text(encoding="utf-8")
    assert "Epidemiologic interpretation guidance" in summary_text
    assert "Variance inflation factors" in summary_text


def test_ecological_regression_excludes_optional_predictor_when_coverage_is_low(tmp_path: Path) -> None:
    frame = _synthetic_clean_panel()
    frame.loc[frame.index % 3 != 0, "skilled_birth_attendance"] = np.nan

    result = run_ecological_regression(
        clean_panel=frame,
        output_dir=tmp_path,
        minimum_observations=30,
        optional_predictor_min_coverage=0.75,
    )

    assert OPTIONAL_PREDICTOR not in result.predictors_used
    summary_table = pd.read_csv(result.artifacts.model_summary_csv_path)
    optional_note = str(summary_table["optional_predictor_note"].iloc[0])
    assert "excluded" in optional_note


def test_ecological_regression_raises_when_required_columns_missing(tmp_path: Path) -> None:
    frame = _synthetic_clean_panel().drop(columns=["urban_population_pct"])
    with pytest.raises(ModelingError):
        run_ecological_regression(
            clean_panel=frame,
            output_dir=tmp_path,
            minimum_observations=10,
            optional_predictor_min_coverage=0.5,
        )


def test_ecological_regression_enforces_minimum_observations(tmp_path: Path) -> None:
    frame = _synthetic_clean_panel(n_countries=2, start_year=2020, end_year=2021)
    with pytest.raises(ModelingError):
        run_ecological_regression(
            clean_panel=frame,
            output_dir=tmp_path,
            minimum_observations=100,
            optional_predictor_min_coverage=0.5,
        )


def test_specification_comparison_covers_all_specifications() -> None:
    """
    The robustness table must report every predictor under every specification.

    This is the artifact that lets a reader check whether a coefficient survives
    within-country identification or only exists in the cross-section.
    """
    import pandas as pd

    from maternal_mortality_dashboard.modeling.ecological_regression import (
        SPEC_COUNTRY_FE,
        SPEC_POOLED_CLUSTER,
        SPEC_POOLED_HC3,
        SPEC_TWOWAY_FE,
    )

    path = Path("data/processed/ecological_regression_specification_comparison.csv")
    if not path.exists():
        pytest.skip("pipeline artifacts not built; run scripts/run_pipeline.py")

    frame = pd.read_csv(path)
    expected = {SPEC_POOLED_HC3, SPEC_POOLED_CLUSTER, SPEC_COUNTRY_FE, SPEC_TWOWAY_FE}
    assert expected.issubset(set(frame["specification"]))

    # Pooled HC3 and pooled clustered differ only in their standard errors:
    # identical point estimates, but clustering must never shrink the error.
    hc3 = frame[frame["specification"] == SPEC_POOLED_HC3].set_index("term")
    clustered = frame[frame["specification"] == SPEC_POOLED_CLUSTER].set_index("term")
    for term in hc3.index:
        assert hc3.loc[term, "coefficient"] == pytest.approx(
            clustered.loc[term, "coefficient"]
        )
        assert clustered.loc[term, "std_error"] >= hc3.loc[term, "std_error"] * 0.99


def test_fixed_effects_absorb_between_country_variation() -> None:
    """Country fixed effects should raise R-squared substantially: MMR is persistent."""
    import pandas as pd

    from maternal_mortality_dashboard.modeling.ecological_regression import (
        SPEC_COUNTRY_FE,
        SPEC_POOLED_CLUSTER,
    )

    path = Path("data/processed/ecological_regression_specification_comparison.csv")
    if not path.exists():
        pytest.skip("pipeline artifacts not built; run scripts/run_pipeline.py")

    frame = pd.read_csv(path)
    pooled_r2 = frame[frame["specification"] == SPEC_POOLED_CLUSTER]["r_squared"].iloc[0]
    fe_r2 = frame[frame["specification"] == SPEC_COUNTRY_FE]["r_squared"].iloc[0]
    assert fe_r2 > pooled_r2
