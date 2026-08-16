from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from maternal_mortality_dashboard.exceptions import ModelingError

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

BASE_PREDICTORS = [
    "log_gdp_per_capita",
    "female_literacy_rate",
    "health_expenditure_per_capita",
    "urban_population_pct",
]
OPTIONAL_PREDICTOR = "skilled_birth_attendance"


# Specification labels used in the robustness comparison.
SPEC_POOLED_HC3 = "pooled_hc3"
SPEC_POOLED_CLUSTER = "pooled_cluster_country"
SPEC_COUNTRY_FE = "country_fe_cluster"
SPEC_TWOWAY_FE = "country_year_fe_cluster"


@dataclass(frozen=True)
class EcologicalRegressionArtifacts:
    regression_dataset_path: Path
    coefficients_path: Path
    model_summary_csv_path: Path
    model_summary_text_path: Path
    vif_path: Path
    residuals_vs_fitted_plot_path: Path
    qq_plot_path: Path
    specification_comparison_path: Path
    within_country_coefficients_path: Path


@dataclass(frozen=True)
class EcologicalRegressionResult:
    artifacts: EcologicalRegressionArtifacts
    predictors_used: list[str]
    n_observations: int
    r_squared: float
    adj_r_squared: float


def _prepare_regression_frame(clean_panel: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "country_iso3",
        "year",
        "mmr",
        "gdp_per_capita",
        "female_literacy_rate",
        "health_expenditure_per_capita",
        "urban_population_pct",
    ]
    missing = set(required_columns).difference(clean_panel.columns)
    if missing:
        raise ModelingError(f"Missing required regression columns: {sorted(missing)}")

    columns = required_columns.copy()
    if OPTIONAL_PREDICTOR in clean_panel.columns:
        columns.append(OPTIONAL_PREDICTOR)

    regression_frame = clean_panel[columns].copy()
    numeric_columns = [
        "mmr",
        "gdp_per_capita",
        "female_literacy_rate",
        "health_expenditure_per_capita",
        "urban_population_pct",
    ]
    if OPTIONAL_PREDICTOR in regression_frame.columns:
        numeric_columns.append(OPTIONAL_PREDICTOR)

    for column in numeric_columns:
        regression_frame[column] = pd.to_numeric(regression_frame[column], errors="coerce")

    regression_frame = regression_frame.loc[
        (regression_frame["mmr"] > 0) & (regression_frame["gdp_per_capita"] > 0)
    ].copy()
    regression_frame["log_mmr"] = np.log(regression_frame["mmr"])
    regression_frame["log_gdp_per_capita"] = np.log(regression_frame["gdp_per_capita"])
    regression_frame = regression_frame.replace([np.inf, -np.inf], np.nan)
    return regression_frame


def _resolve_model_specification(
    regression_frame: pd.DataFrame,
    minimum_observations: int,
    optional_predictor_min_coverage: float,
) -> tuple[pd.DataFrame, list[str], str]:
    model_frame = regression_frame.dropna(subset=["log_mmr"] + BASE_PREDICTORS).copy()
    predictors = BASE_PREDICTORS.copy()
    optional_note = "skilled_birth_attendance excluded: indicator unavailable in cleaned panel."

    if OPTIONAL_PREDICTOR in regression_frame.columns:
        non_null_coverage = float(model_frame[OPTIONAL_PREDICTOR].notna().mean())
        with_optional = model_frame.dropna(subset=[OPTIONAL_PREDICTOR]).copy()
        if non_null_coverage >= optional_predictor_min_coverage and len(with_optional) >= minimum_observations:
            model_frame = with_optional
            predictors.append(OPTIONAL_PREDICTOR)
            optional_note = (
                "skilled_birth_attendance included: "
                f"{non_null_coverage:.1%} coverage met threshold {optional_predictor_min_coverage:.1%}."
            )
        else:
            optional_note = (
                "skilled_birth_attendance excluded: "
                f"{non_null_coverage:.1%} coverage below threshold {optional_predictor_min_coverage:.1%} "
                f"or insufficient complete rows ({len(with_optional)})."
            )

    if len(model_frame) < minimum_observations:
        raise ModelingError(
            f"Insufficient complete observations for ecological regression: {len(model_frame)} "
            f"(minimum required: {minimum_observations})"
        )

    return model_frame, predictors, optional_note


def _fit_specifications(
    model_frame: pd.DataFrame, predictors: list[str]
) -> dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Fit the same predictors under four error/identification assumptions.

    Why this exists: a pooled OLS with HC3 errors treats 2,823 country-years as
    independent observations. They are not - each country contributes up to 23
    highly autocorrelated rows, so HC3 (which corrects heteroskedasticity only)
    understates the standard errors badly and produces implausible p-values.

    The four specifications separate two distinct questions:

    * ``pooled_hc3`` / ``pooled_cluster_country`` - identical coefficients,
      different uncertainty. The cluster version is the honest one.
    * ``country_fe_cluster`` - adds country dummies, so coefficients are
      identified from *within-country change over time* rather than from
      differences between rich and poor countries. This is the specification
      that matches a counterfactual like "if this country raised skilled birth
      attendance", which is what the dashboard's scenario panel asks.
    * ``country_year_fe_cluster`` - adds year dummies too, absorbing global
      shocks common to all countries (the 2020-21 pandemic being the obvious
      one).

    Coefficients that survive fixed effects are far more defensible than
    coefficients that only appear in the cross-section.
    """
    groups = model_frame["country_iso3"]
    outcome = model_frame["log_mmr"]
    cluster = {"groups": groups}

    base = sm.add_constant(model_frame[predictors], has_constant="add")
    fitted: dict[str, sm.regression.linear_model.RegressionResultsWrapper] = {}
    fitted[SPEC_POOLED_HC3] = sm.OLS(outcome, base).fit(cov_type="HC3")
    fitted[SPEC_POOLED_CLUSTER] = sm.OLS(outcome, base).fit(
        cov_type="cluster", cov_kwds=cluster
    )

    country_dummies = pd.get_dummies(groups, prefix="cty", drop_first=True, dtype=float)
    with_country = pd.concat([base, country_dummies.set_index(base.index)], axis=1)
    fitted[SPEC_COUNTRY_FE] = sm.OLS(outcome, with_country).fit(
        cov_type="cluster", cov_kwds=cluster
    )

    year_dummies = pd.get_dummies(
        model_frame["year"], prefix="yr", drop_first=True, dtype=float
    )
    with_both = pd.concat([with_country, year_dummies.set_index(base.index)], axis=1)
    fitted[SPEC_TWOWAY_FE] = sm.OLS(outcome, with_both).fit(
        cov_type="cluster", cov_kwds=cluster
    )
    return fitted


def _specification_comparison(
    fitted: dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    predictors: list[str],
) -> pd.DataFrame:
    """One row per (specification, predictor) so the sensitivity is auditable."""
    rows: list[dict[str, object]] = []
    for label, results in fitted.items():
        for term in predictors:
            if term not in results.params.index:
                continue
            rows.append(
                {
                    "specification": label,
                    "term": term,
                    "coefficient": float(results.params[term]),
                    "std_error": float(results.bse[term]),
                    "p_value": float(results.pvalues[term]),
                    "ci_lower_95": float(results.conf_int().loc[term, 0]),
                    "ci_upper_95": float(results.conf_int().loc[term, 1]),
                    "significant_at_05": bool(results.pvalues[term] < 0.05),
                    "n_observations": int(results.nobs),
                    "r_squared": float(results.rsquared),
                }
            )
    return pd.DataFrame(rows)


def _compute_vif_table(design_matrix: pd.DataFrame) -> pd.DataFrame:
    vif_rows: list[dict[str, float | str]] = []
    values = design_matrix.to_numpy(dtype=float)
    for column_index, column_name in enumerate(design_matrix.columns):
        try:
            vif_value = float(variance_inflation_factor(values, column_index))
        except Exception:
            vif_value = float("inf")

        vif_rows.append(
            {
                "variable": column_name,
                "vif": vif_value,
                "high_multicollinearity_flag": bool(vif_value >= 5.0) if np.isfinite(vif_value) else True,
            }
        )

    return pd.DataFrame(vif_rows)


def _epidemiologic_interpretation(term: str, coefficient: float) -> str:
    direction = "higher" if coefficient > 0 else "lower"
    if term == "const":
        return (
            "Baseline expected log-MMR when all predictors are zero; mainly a model centering term "
            "with limited epidemiologic interpretability."
        )
    if term == "log_gdp_per_capita":
        return (
            f"A 1% higher GDP per capita is associated with an estimated {abs(coefficient):.2f}% {direction} "
            "maternal mortality ratio, holding other predictors constant."
        )
    if term in {"female_literacy_rate", "urban_population_pct", "skilled_birth_attendance"}:
        return (
            f"A 1 percentage-point increase in {term} is associated with an estimated "
            f"{abs(100 * coefficient):.2f}% {direction} maternal mortality ratio, adjusted for other predictors."
        )
    if term == "health_expenditure_per_capita":
        return (
            f"Each additional US$100 in health expenditure per capita is associated with an estimated "
            f"{abs(10000 * coefficient):.2f}% {direction} maternal mortality ratio, conditional on covariates."
        )
    return (
        f"A one-unit increase in {term} is associated with an estimated {abs(100 * coefficient):.2f}% "
        f"{direction} maternal mortality ratio."
    )


def _save_diagnostic_plots(results: sm.regression.linear_model.RegressionResultsWrapper, artifacts: EcologicalRegressionArtifacts) -> None:
    residuals = np.asarray(results.resid)
    fitted_values = np.asarray(results.fittedvalues)

    residuals_fig, residuals_ax = plt.subplots(figsize=(8, 6))
    residuals_ax.scatter(fitted_values, residuals, alpha=0.65, color="#1f77b4")
    residuals_ax.axhline(0.0, color="#d62728", linewidth=1.2, linestyle="--")
    residuals_ax.set_title("Residuals vs Fitted (Ecological OLS)")
    residuals_ax.set_xlabel("Fitted log(MMR)")
    residuals_ax.set_ylabel("Residuals")
    residuals_fig.tight_layout()
    residuals_fig.savefig(artifacts.residuals_vs_fitted_plot_path, dpi=180)
    plt.close(residuals_fig)

    qq_fig = plt.figure(figsize=(8, 6))
    qq_ax = qq_fig.add_subplot(111)
    sm.qqplot(residuals, line="45", ax=qq_ax)
    qq_ax.set_title("Q-Q Plot of Regression Residuals")
    qq_fig.tight_layout()
    qq_fig.savefig(artifacts.qq_plot_path, dpi=180)
    plt.close(qq_fig)


def _build_artifact_paths(output_dir: Path) -> EcologicalRegressionArtifacts:
    return EcologicalRegressionArtifacts(
        regression_dataset_path=output_dir / "ecological_regression_dataset.parquet",
        coefficients_path=output_dir / "ecological_regression_coefficients.csv",
        model_summary_csv_path=output_dir / "ecological_regression_model_summary.csv",
        model_summary_text_path=output_dir / "ecological_regression_model_summary.txt",
        vif_path=output_dir / "ecological_regression_vif.csv",
        residuals_vs_fitted_plot_path=output_dir / "ecological_regression_residuals_vs_fitted.png",
        qq_plot_path=output_dir / "ecological_regression_residuals_qq_plot.png",
        specification_comparison_path=output_dir / "ecological_regression_specification_comparison.csv",
        within_country_coefficients_path=output_dir / "ecological_regression_within_country_coefficients.csv",
    )


def run_ecological_regression(
    clean_panel: pd.DataFrame,
    output_dir: Path,
    minimum_observations: int,
    optional_predictor_min_coverage: float,
) -> EcologicalRegressionResult:
    """
    Fit an ecological regression for log(MMR) with robust (HC3) standard errors.

    Epidemiologic framing:
    - Unit of analysis is country-year.
    - Coefficients represent population-level associations and should not be interpreted as individual-level causal effects.
    - Log-transformed outcome supports proportional interpretation of effect sizes.
    """

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = _build_artifact_paths(output_dir)

        regression_frame = _prepare_regression_frame(clean_panel)
        model_frame, predictors, optional_note = _resolve_model_specification(
            regression_frame=regression_frame,
            minimum_observations=minimum_observations,
            optional_predictor_min_coverage=optional_predictor_min_coverage,
        )

        design_matrix = sm.add_constant(model_frame[predictors], has_constant="add")

        # Fit every specification, then report the cluster-robust pooled model as
        # the headline: it has the same coefficients as the HC3 version but
        # standard errors that account for repeated observations per country.
        specifications = _fit_specifications(model_frame, predictors)
        results = specifications[SPEC_POOLED_CLUSTER]
        comparison = _specification_comparison(specifications, predictors)

        confidence_intervals = results.conf_int()
        coefficients = pd.DataFrame(
            {
                "term": results.params.index,
                "coefficient": results.params.values,
                "robust_std_error": results.bse.values,
                "z_or_t_stat": results.tvalues.values,
                "p_value": results.pvalues.values,
                "ci_lower_95": confidence_intervals[0].values,
                "ci_upper_95": confidence_intervals[1].values,
            }
        )
        coefficients["epidemiologic_interpretation"] = coefficients.apply(
            lambda row: _epidemiologic_interpretation(
                term=str(row["term"]),
                coefficient=float(row["coefficient"]),
            ),
            axis=1,
        )

        vif_table = _compute_vif_table(design_matrix)

        summary_table = pd.DataFrame(
            [
                {
                    "outcome": "log_mmr",
                    "model_type": "OLS",
                    "covariance_estimator": "cluster(country_iso3)",
                    "n_observations": int(results.nobs),
                    "n_countries": int(model_frame["country_iso3"].nunique()),
                    "year_min": int(model_frame["year"].min()),
                    "year_max": int(model_frame["year"].max()),
                    "r_squared": float(results.rsquared),
                    "adj_r_squared": float(results.rsquared_adj),
                    "aic": float(results.aic),
                    "bic": float(results.bic),
                    "f_statistic": float(results.fvalue),
                    "f_pvalue": float(results.f_pvalue),
                    "predictors_used": ";".join(predictors),
                    "optional_predictor_note": optional_note,
                }
            ]
        )

        model_frame_with_diagnostics = model_frame.copy()
        model_frame_with_diagnostics["fitted_log_mmr"] = results.fittedvalues
        model_frame_with_diagnostics["residual"] = results.resid

        model_summary_text = (
            "Ecological regression for maternal mortality inequality\n"
            "Outcome: log-transformed maternal mortality ratio (log_mmr)\n"
            f"Predictors used: {', '.join(predictors)}\n"
            f"Optional predictor rule: {optional_note}\n\n"
            "Epidemiologic interpretation guidance:\n"
            "- Associations are ecological (country-year) and may reflect structural confounding.\n"
            "- Coefficients describe adjusted average directional relationships, not causal effects.\n"
            "- For log-log terms, coefficients are elasticities.\n"
            "- For log-linear terms, 100 x beta approximates percent change in MMR per unit change.\n\n"
            "Model summary:\n"
            f"{results.summary().as_text()}\n\n"
            "Variance inflation factors (multicollinearity check):\n"
            f"{vif_table.to_string(index=False)}\n"
        )

        model_frame_with_diagnostics.to_parquet(artifacts.regression_dataset_path, index=False)
        coefficients.to_csv(artifacts.coefficients_path, index=False)
        comparison.to_csv(artifacts.specification_comparison_path, index=False)

        # Within-country coefficients drive the dashboard's scenario panel: asking
        # "what if this country changed X" is a within-country counterfactual, so
        # between-country coefficients would be the ecological fallacy in action.
        within = specifications[SPEC_TWOWAY_FE]
        within_frame = pd.DataFrame(
            {
                "term": ["const"] + predictors,
                "coefficient": [float(within.params["const"])]
                + [float(within.params[p]) for p in predictors],
                "std_error": [float(within.bse["const"])]
                + [float(within.bse[p]) for p in predictors],
                "p_value": [float(within.pvalues["const"])]
                + [float(within.pvalues[p]) for p in predictors],
            }
        )
        within_frame.to_csv(artifacts.within_country_coefficients_path, index=False)
        summary_table.to_csv(artifacts.model_summary_csv_path, index=False)
        artifacts.model_summary_text_path.write_text(model_summary_text, encoding="utf-8")
        vif_table.to_csv(artifacts.vif_path, index=False)
        _save_diagnostic_plots(results, artifacts)

        logger.info(
            "Ecological regression completed with %s observations and predictors: %s",
            int(results.nobs),
            predictors,
        )

        return EcologicalRegressionResult(
            artifacts=artifacts,
            predictors_used=predictors,
            n_observations=int(results.nobs),
            r_squared=float(results.rsquared),
            adj_r_squared=float(results.rsquared_adj),
        )
    except Exception as exc:
        raise ModelingError("Failed to run ecological regression modeling workflow") from exc
