from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

METRIC_LABELS = {
    "mmr": "Maternal Mortality Ratio (per 100,000 live births)",
    "gdp_per_capita": "GDP Per Capita (current US$)",
    "female_secondary_completion": "Female Secondary Completion (%)",
}


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data available for the selected filters.",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        template="plotly_white",
    )
    return figure


def choropleth_figure(df: pd.DataFrame, metric: str, year: int) -> go.Figure:
    if df.empty or metric not in df.columns:
        return _empty_figure(f"{METRIC_LABELS.get(metric, metric)} in {year}")

    plotting_df = df.loc[df[metric].notna()].copy()
    if plotting_df.empty:
        return _empty_figure(f"{METRIC_LABELS.get(metric, metric)} in {year}")

    figure = px.choropleth(
        plotting_df,
        locations="country_iso3",
        color=metric,
        hover_name="country_name",
        hover_data={"region": True, "income_level": True},
        color_continuous_scale="YlOrRd",
        projection="natural earth",
        title=f"{METRIC_LABELS.get(metric, metric)} in {year}",
    )
    figure.update_layout(template="plotly_white", margin={"l": 0, "r": 0, "t": 48, "b": 0})
    return figure


def inequality_trend_figure(yearly_inequality: pd.DataFrame, selected_year: int) -> go.Figure:
    filtered = yearly_inequality.loc[yearly_inequality["year"] <= selected_year].copy()
    if filtered.empty:
        return _empty_figure("Global Maternal Mortality Inequality Trend")

    figure = px.line(
        filtered,
        x="year",
        y="gini_mmr",
        markers=True,
        title="Global Maternal Mortality Inequality (Gini) Over Time",
    )
    figure.update_layout(template="plotly_white")
    figure.update_yaxes(title="Gini coefficient")
    figure.update_xaxes(title="Year")
    return figure


def mmr_vs_gdp_figure(df: pd.DataFrame, year: int) -> go.Figure:
    if df.empty:
        return _empty_figure(f"MMR vs GDP in {year}")

    plotting_df = df.loc[df["mmr"].notna() & df["gdp_per_capita"].notna()].copy()
    if plotting_df.empty:
        return _empty_figure(f"MMR vs GDP in {year}")

    size_col = "female_secondary_completion"
    if plotting_df[size_col].notna().any():
        size_values = plotting_df[size_col].fillna(float(plotting_df[size_col].median()))
    else:
        size_values = pd.Series(50.0, index=plotting_df.index)
    plotting_df[size_col] = size_values.clip(lower=1.0)

    figure = px.scatter(
        plotting_df,
        x="gdp_per_capita",
        y="mmr",
        color="income_level",
        hover_name="country_name",
        size=size_col,
        size_max=24,
        title=f"MMR vs GDP Per Capita in {year}",
        labels={
            "gdp_per_capita": "GDP Per Capita (current US$)",
            "mmr": "Maternal Mortality Ratio",
            "income_level": "Income Level",
            "female_secondary_completion": "Female Secondary Completion (%)",
        },
    )
    figure.update_layout(template="plotly_white")
    figure.update_yaxes(type="log")
    return figure


def income_group_bar_figure(df: pd.DataFrame, year: int) -> go.Figure:
    filtered = df.loc[df["year"] == year].copy()
    filtered = filtered.loc[filtered["mmr_mean"].notna() & filtered["income_level"].notna()].copy()
    if filtered.empty:
        return _empty_figure(f"Income Group Comparison in {year}")

    figure = px.bar(
        filtered.sort_values("mmr_mean", ascending=False),
        x="income_level",
        y="mmr_mean",
        color="income_level",
        title=f"Average MMR by Income Group in {year}",
        labels={
            "income_level": "Income Group",
            "mmr_mean": "Average Maternal Mortality Ratio",
        },
    )
    figure.update_layout(showlegend=False, template="plotly_white")
    return figure


def country_trend_figure(df: pd.DataFrame, country_iso3: str, metric: str) -> go.Figure:
    if metric not in df.columns:
        return _empty_figure("Country Trend")

    filtered = df.loc[df["country_iso3"] == country_iso3].copy()
    if filtered.empty:
        return _empty_figure("Country Trend")

    filtered = filtered.sort_values("year").reset_index(drop=True)
    country_name = str(filtered["country_name"].iloc[0])
    label = METRIC_LABELS.get(metric, metric)
    plotting_metric = metric
    plotting_df = filtered.loc[filtered[metric].notna(), ["year", metric]].copy()

    # Fall back to MMR when selected metric is unavailable for the country.
    if plotting_df.empty and metric != "mmr" and "mmr" in filtered.columns:
        mmr_df = filtered.loc[filtered["mmr"].notna(), ["year", "mmr"]].copy()
        if not mmr_df.empty:
            plotting_metric = "mmr"
            plotting_df = mmr_df
            label = f"{METRIC_LABELS.get(metric, metric)} unavailable; showing MMR"

    if plotting_df.empty:
        return _empty_figure(f"{country_name}: {label}")

    figure = px.line(
        plotting_df,
        x="year",
        y=plotting_metric,
        markers=True,
        title=f"{country_name}: {label}",
    )
    figure.update_layout(template="plotly_white")
    figure.update_xaxes(title="Year")
    figure.update_yaxes(title=label)
    return figure
