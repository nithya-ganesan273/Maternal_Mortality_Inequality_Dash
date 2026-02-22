from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from dash import Dash, Input, Output

from maternal_mortality_dashboard.dashboard.data_access import DashboardDatasets
from maternal_mortality_dashboard.dashboard.figures import (
    METRIC_LABELS,
    choropleth_figure,
    country_trend_figure,
    income_group_bar_figure,
    inequality_trend_figure,
    mmr_vs_gdp_figure,
)
from maternal_mortality_dashboard.dashboard.scenario_model import predict_adjusted_mmr


def _country_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    countries = (
        df[["country_iso3", "country_name"]]
        .drop_duplicates()
        .sort_values("country_name")
        .to_dict(orient="records")
    )
    return [{"label": row["country_name"], "value": row["country_iso3"]} for row in countries]


def _year_frame(
    country_year_df: pd.DataFrame,
    selected_year: int,
    selected_regions: list[str] | None,
) -> pd.DataFrame:
    frame = country_year_df.loc[country_year_df["year"] == selected_year].copy()
    if selected_regions:
        frame = frame.loc[frame["region"].isin(selected_regions)].copy()
    return frame


def _select_country_row(frame: pd.DataFrame, selected_country: str | None) -> pd.Series | None:
    if not selected_country:
        return None

    country_rows = frame.loc[frame["country_iso3"] == selected_country]
    if country_rows.empty:
        return None
    return country_rows.iloc[0]


def _median_or_default(frame: pd.DataFrame, column: str, default: float) -> float:
    if column not in frame.columns:
        return default
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.median())


def _format_mmr(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.1f} per 100,000"


def _format_percent(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:+.1f}%"


def register_callbacks(app: Dash, datasets: DashboardDatasets) -> None:
    country_year_df = datasets.country_year.copy()
    yearly_ineq_df = datasets.yearly_inequality.copy()
    income_group_df = datasets.income_group_summary.copy()
    regression_coefficients_df = datasets.regression_coefficients.copy()
    regression_summary_df = datasets.regression_model_summary.copy()

    @app.callback(
        Output("map-graph", "figure"),
        Output("inequality-graph", "figure"),
        Output("scatter-graph", "figure"),
        Output("income-summary-graph", "figure"),
        Output("country-dropdown", "options"),
        Output("country-dropdown", "value"),
        Output("kpi-total-countries", "children"),
        Output("kpi-global-median", "children"),
        Output("kpi-global-gini", "children"),
        Input("year-slider", "value"),
        Input("region-dropdown", "value"),
        Input("metric-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("map-graph", "clickData"),
    )
    def update_overview(
        selected_year: int,
        selected_regions: list[str] | None,
        selected_metric: str,
        selected_country: str | None,
        map_click_data: dict[str, Any] | None,
    ) -> tuple[Any, Any, Any, Any, list[dict[str, str]], str | None, str, str, str]:
        if selected_metric not in METRIC_LABELS:
            selected_metric = "mmr"

        year_filtered = _year_frame(country_year_df, selected_year, selected_regions)
        options = _country_options(year_filtered)
        option_values = {option["value"] for option in options}
        clicked_country = None
        if map_click_data and isinstance(map_click_data, dict):
            points = map_click_data.get("points")
            if points and isinstance(points, list):
                clicked_country = points[0].get("location")

        if clicked_country in option_values:
            resolved_country = str(clicked_country)
        elif selected_country in option_values:
            resolved_country = selected_country
        elif options:
            resolved_country = options[0]["value"]
        else:
            resolved_country = None

        map_fig = choropleth_figure(year_filtered, selected_metric, selected_year)
        inequality_fig = inequality_trend_figure(yearly_ineq_df, selected_year)
        scatter_fig = mmr_vs_gdp_figure(year_filtered, selected_year)
        income_fig = income_group_bar_figure(income_group_df, selected_year)

        kpi_total = f"{year_filtered['country_iso3'].nunique():,}"
        yearly_row = yearly_ineq_df.loc[yearly_ineq_df["year"] == selected_year]
        if yearly_row.empty:
            kpi_median = "N/A"
            kpi_gini = "N/A"
        else:
            median_value = float(yearly_row["mmr_median"].iloc[0])
            gini_value = float(yearly_row["gini_mmr"].iloc[0])
            kpi_median = f"{median_value:,.1f}"
            kpi_gini = f"{gini_value:.3f}"

        return (
            map_fig,
            inequality_fig,
            scatter_fig,
            income_fig,
            options,
            resolved_country,
            kpi_total,
            kpi_median,
            kpi_gini,
        )

    @app.callback(
        Output("scenario-literacy-slider", "value"),
        Output("scenario-health-exp-slider", "value"),
        Output("scenario-skilled-birth-slider", "value"),
        Input("year-slider", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
    )
    def update_scenario_slider_defaults(
        selected_year: int,
        selected_regions: list[str] | None,
        selected_country: str | None,
    ) -> tuple[float, float, float]:
        frame = _year_frame(country_year_df, selected_year, selected_regions)
        row = _select_country_row(frame, selected_country)

        fallback_literacy = _median_or_default(frame, "female_literacy_rate", 50.0)
        fallback_health = _median_or_default(frame, "health_expenditure_per_capita", 1000.0)
        fallback_skilled = _median_or_default(frame, "skilled_birth_attendance", 60.0)

        if row is None:
            return fallback_literacy, fallback_health, fallback_skilled

        literacy = _median_or_default(pd.DataFrame([row]), "female_literacy_rate", fallback_literacy)
        health = _median_or_default(pd.DataFrame([row]), "health_expenditure_per_capita", fallback_health)
        skilled = _median_or_default(pd.DataFrame([row]), "skilled_birth_attendance", fallback_skilled)

        literacy = float(np.clip(literacy, 0.0, 100.0))
        skilled = float(np.clip(skilled, 0.0, 100.0))
        health = float(max(0.0, health))
        return literacy, health, skilled

    @app.callback(
        Output("scenario-baseline-mmr", "children"),
        Output("scenario-adjusted-mmr", "children"),
        Output("scenario-percent-change", "children"),
        Output("scenario-model-note", "children"),
        Input("year-slider", "value"),
        Input("region-dropdown", "value"),
        Input("country-dropdown", "value"),
        Input("scenario-literacy-slider", "value"),
        Input("scenario-health-exp-slider", "value"),
        Input("scenario-skilled-birth-slider", "value"),
    )
    def update_scenario_prediction(
        selected_year: int,
        selected_regions: list[str] | None,
        selected_country: str | None,
        literacy_value: float | None,
        health_expenditure_value: float | None,
        skilled_birth_value: float | None,
    ) -> tuple[str, str, str, str]:
        frame = _year_frame(country_year_df, selected_year, selected_regions)
        row = _select_country_row(frame, selected_country)

        fallback_literacy = _median_or_default(frame, "female_literacy_rate", 50.0)
        fallback_health = _median_or_default(frame, "health_expenditure_per_capita", 1000.0)
        fallback_skilled = _median_or_default(frame, "skilled_birth_attendance", 60.0)

        prediction = predict_adjusted_mmr(
            row=row,
            coefficient_table=regression_coefficients_df,
            literacy_value=literacy_value,
            health_expenditure_value=health_expenditure_value,
            skilled_birth_value=skilled_birth_value,
            fallback_literacy=fallback_literacy,
            fallback_health_expenditure=fallback_health,
            fallback_skilled_birth=fallback_skilled,
        )

        summary_note = ""
        if "optional_predictor_note" in regression_summary_df.columns and not regression_summary_df.empty:
            summary_note = str(regression_summary_df["optional_predictor_note"].iloc[0])

        detail_note = prediction.note
        if summary_note:
            detail_note = f"{detail_note} {summary_note}".strip()

        return (
            _format_mmr(prediction.baseline_mmr),
            _format_mmr(prediction.adjusted_mmr),
            _format_percent(prediction.percent_change),
            detail_note,
        )

    @app.callback(
        Output("country-trend-graph", "figure"),
        Input("country-dropdown", "value"),
        Input("metric-dropdown", "value"),
    )
    def update_country_trend(selected_country: str | None, selected_metric: str) -> Any:
        if not selected_country:
            return country_trend_figure(country_year_df.iloc[0:0], "", "mmr")
        if selected_metric not in METRIC_LABELS:
            selected_metric = "mmr"
        return country_trend_figure(country_year_df, selected_country, selected_metric)
