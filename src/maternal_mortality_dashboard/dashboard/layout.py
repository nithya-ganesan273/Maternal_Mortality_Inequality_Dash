from __future__ import annotations

from typing import Any

import pandas as pd
from dash import dcc, html

from maternal_mortality_dashboard.dashboard.figures import METRIC_LABELS


def _kpi_card(title: str, component_id: str) -> html.Div:
    return html.Div(
        children=[
            html.P(title, style={"margin": "0 0 6px 0", "fontSize": "0.9rem", "color": "#586069"}),
            html.H3(id=component_id, style={"margin": 0}),
        ],
        style={
            "backgroundColor": "#F6F8FA",
            "border": "1px solid #D0D7DE",
            "padding": "12px 16px",
            "borderRadius": "10px",
            "minWidth": "220px",
        },
    )


def _country_options(df: pd.DataFrame) -> list[dict[str, Any]]:
    countries = (
        df[["country_iso3", "country_name"]]
        .drop_duplicates()
        .sort_values("country_name")
        .to_dict(orient="records")
    )
    return [{"label": row["country_name"], "value": row["country_iso3"]} for row in countries]


def _methods_section(title: str, body: str) -> html.Div:
    return html.Div(
        children=[
            html.H3(title, style={"marginBottom": "8px"}),
            html.P(
                body,
                style={
                    "marginTop": 0,
                    "lineHeight": "1.7",
                    "color": "#1F2328",
                    "fontSize": "1rem",
                },
            ),
        ],
        style={
            "backgroundColor": "#FFFFFF",
            "border": "1px solid #D0D7DE",
            "borderRadius": "12px",
            "padding": "16px 18px",
        },
    )


def _scenario_result_card(title: str, component_id: str) -> html.Div:
    return html.Div(
        children=[
            html.P(title, style={"margin": "0 0 6px 0", "fontSize": "0.9rem", "color": "#586069"}),
            html.H3(id=component_id, style={"margin": 0}),
        ],
        style={
            "backgroundColor": "#FFFFFF",
            "border": "1px solid #D0D7DE",
            "padding": "12px 16px",
            "borderRadius": "10px",
            "minWidth": "220px",
        },
    )


def _methods_limitations_content() -> html.Div:
    return html.Div(
        style={"display": "grid", "gap": "12px", "marginTop": "14px"},
        children=[
            _methods_section(
                "Ecological Study Design",
                (
                    "This dashboard is based on an ecological panel design in which country-year observations are "
                    "the analytic units. The objective is to characterize cross-national patterns in maternal "
                    "mortality inequality and associated structural determinants at the population level."
                ),
            ),
            _methods_section(
                "Rationale for Log Transformation",
                (
                    "Maternal mortality ratios are right-skewed across countries and years. A logarithmic "
                    "transformation is used to improve scale comparability, reduce the influence of extreme "
                    "values, and support interpretation of model coefficients in proportional terms."
                ),
            ),
            _methods_section(
                "Non-Causal Interpretation",
                (
                    "Estimated associations should be interpreted as descriptive and inferentially adjusted "
                    "relationships rather than causal effects. The ecological structure does not permit individual-"
                    "level causal inference, and observed coefficients may reflect concurrent social, economic, and "
                    "health-system dynamics."
                ),
            ),
            _methods_section(
                "Potential Confounding",
                (
                    "Potential confounding may arise from unmeasured governance conditions, quality of obstetric "
                    "care, fertility dynamics, conflict, and temporal policy changes. Although multivariable models "
                    "address measured covariates, residual confounding remains plausible and should be considered "
                    "when interpreting between-country contrasts."
                ),
            ),
            _methods_section(
                "Data Limitations",
                (
                    "Indicators are harmonized from international reporting systems and may differ in completeness, "
                    "timeliness, and measurement precision across settings. Country comparability can be affected by "
                    "variation in surveillance capacity, registration quality, and methodological revisions over time."
                ),
            ),
            _methods_section(
                "Missing Data Handling",
                (
                    "Missingness is addressed through structured preprocessing. Predictors with plausible temporal "
                    "continuity are interpolated within country time series, whereas optional indicators with limited "
                    "coverage are conditionally excluded from regression specification when completeness thresholds are "
                    "not met. These decisions prioritize analytic stability while preserving transparency regarding "
                    "data gaps."
                ),
            ),
        ],
    )


def _dashboard_content(
    years: list[int],
    region_options: list[str],
    initial_country_options: list[dict[str, str]],
    default_country: str | None,
) -> html.Div:
    min_year = min(years)
    max_year = max(years)
    slider_marks = {year: str(year) for year in years[:: max(1, len(years) // 8)] + [max_year]}

    return html.Div(
        children=[
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
                    "gap": "12px",
                    "marginTop": "12px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        children=[
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="metric-dropdown",
                                options=[{"label": label, "value": key} for key, label in METRIC_LABELS.items()],
                                value="mmr",
                                clearable=False,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Region"),
                            dcc.Dropdown(
                                id="region-dropdown",
                                options=[{"label": region, "value": region} for region in region_options],
                                multi=True,
                                placeholder="All regions",
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label("Country"),
                            dcc.Dropdown(
                                id="country-dropdown",
                                options=initial_country_options,
                                value=default_country,
                                placeholder="Select a country",
                            ),
                        ]
                    ),
                ],
            ),
            html.Div(
                style={"marginBottom": "20px"},
                children=[
                    html.Label("Year"),
                    dcc.Slider(
                        id="year-slider",
                        min=min_year,
                        max=max_year,
                        step=1,
                        value=max_year,
                        marks=slider_marks,
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "20px"},
                children=[
                    _kpi_card("Countries in scope", "kpi-total-countries"),
                    _kpi_card("Global median MMR", "kpi-global-median"),
                    _kpi_card("Global MMR Gini", "kpi-global-gini"),
                ],
            ),
            html.Div(
                style={
                    "backgroundColor": "#F6F8FA",
                    "border": "1px solid #D0D7DE",
                    "borderRadius": "12px",
                    "padding": "16px",
                    "marginBottom": "16px",
                },
                children=[
                    html.Div(
                        "Ecological model — non-causal.",
                        style={
                            "display": "inline-block",
                            "backgroundColor": "#FFF8C5",
                            "border": "1px solid #D4A72C",
                            "padding": "4px 10px",
                            "borderRadius": "999px",
                            "fontWeight": "600",
                            "marginBottom": "10px",
                        },
                    ),
                    html.H3(
                        "Adjusted MMR Scenario Analysis",
                        style={"marginTop": 0, "marginBottom": "8px"},
                    ),
                    html.P(
                        (
                            "Starts from the selected country-year's observed mortality and applies "
                            "within-country regression coefficients to whatever the sliders change. "
                            "This estimates how this country's own mortality would move, rather than "
                            "what a different country with these characteristics would look like."
                        ),
                        style={"marginTop": 0, "color": "#57606A"},
                    ),
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr 1fr",
                            "gap": "12px",
                            "marginBottom": "14px",
                        },
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Female literacy rate (%)"),
                                    dcc.Slider(
                                        id="scenario-literacy-slider",
                                        min=0,
                                        max=100,
                                        step=0.1,
                                        value=50,
                                        marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Health expenditure per capita (US$)"),
                                    dcc.Slider(
                                        id="scenario-health-exp-slider",
                                        min=0,
                                        max=5000,
                                        step=10,
                                        value=1000,
                                        marks={0: "0", 1000: "1,000", 2500: "2,500", 5000: "5,000"},
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Skilled birth attendance (%)"),
                                    dcc.Slider(
                                        id="scenario-skilled-birth-slider",
                                        min=0,
                                        max=100,
                                        step=0.1,
                                        value=50,
                                        marks={0: "0", 25: "25", 50: "50", 75: "75", 100: "100"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
                        children=[
                            # "Observed", not "predicted": the baseline is the country's
                            # actual reported MMR, so the card must not imply a model estimate.
                            _scenario_result_card("Observed MMR (baseline)", "scenario-baseline-mmr"),
                            _scenario_result_card("Adjusted MMR (modelled)", "scenario-adjusted-mmr"),
                            _scenario_result_card("Percent change vs baseline", "scenario-percent-change"),
                        ],
                    ),
                    html.P(
                        id="scenario-model-note",
                        style={"marginTop": "12px", "color": "#57606A", "fontSize": "0.95rem"},
                    ),
                ],
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1.25fr 1fr",
                    "gap": "16px",
                    "marginBottom": "16px",
                },
                children=[
                    dcc.Graph(id="map-graph"),
                    dcc.Graph(id="inequality-graph"),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    dcc.Graph(id="scatter-graph"),
                    dcc.Graph(id="country-trend-graph"),
                ],
            ),
            html.Div(style={"marginTop": "16px"}, children=[dcc.Graph(id="income-summary-graph")]),
        ]
    )


def build_layout(country_year_df: pd.DataFrame, latest_snapshot_df: pd.DataFrame) -> html.Div:
    years = sorted(country_year_df["year"].astype(int).unique().tolist())
    region_options = sorted(country_year_df["region"].dropna().unique().tolist())
    initial_country_options = _country_options(latest_snapshot_df)
    default_country = initial_country_options[0]["value"] if initial_country_options else None

    return html.Div(
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px"},
        children=[
            html.H1("Global Maternal Mortality Inequality Dashboard", style={"marginBottom": "4px"}),
            html.P(
                "Track cross-country inequality in maternal mortality and structural drivers over time.",
                style={"marginTop": 0, "color": "#586069"},
            ),
            dcc.Tabs(
                id="content-tabs",
                value="dashboard-tab",
                children=[
                    dcc.Tab(
                        label="Dashboard",
                        value="dashboard-tab",
                        style={"padding": "10px 14px"},
                        selected_style={
                            "padding": "10px 14px",
                            "fontWeight": "600",
                            "borderBottom": "2px solid #0969DA",
                        },
                        children=[
                            _dashboard_content(
                                years=years,
                                region_options=region_options,
                                initial_country_options=initial_country_options,
                                default_country=default_country,
                            )
                        ],
                    ),
                    dcc.Tab(
                        label="Methods & Limitations",
                        value="methods-tab",
                        style={"padding": "10px 14px"},
                        selected_style={
                            "padding": "10px 14px",
                            "fontWeight": "600",
                            "borderBottom": "2px solid #0969DA",
                        },
                        children=[_methods_limitations_content()],
                    ),
                ],
            ),
        ],
    )
