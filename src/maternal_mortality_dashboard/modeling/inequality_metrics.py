from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from maternal_mortality_dashboard.exceptions import ModelingError

logger = logging.getLogger(__name__)

_CANONICAL_INCOME_RANK = {
    "low income": 1.0,
    "lower middle income": 2.0,
    "middle income": 2.5,
    "upper middle income": 3.0,
    "high income": 4.0,
}


def _canonical_income_group(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    normalized = " ".join(str(value).strip().lower().split())
    if not normalized:
        return None
    if "lower middle income" in normalized:
        return "lower middle income"
    if "upper middle income" in normalized:
        return "upper middle income"
    if "high income" in normalized:
        return "high income"
    if normalized == "middle income" or "middle income" in normalized:
        return "middle income"
    if normalized == "low income":
        return "low income"
    return None


def _gini(values: np.ndarray) -> float:
    cleaned = values[np.isfinite(values)]
    cleaned = cleaned[cleaned >= 0]
    if cleaned.size == 0:
        return float("nan")
    if np.allclose(cleaned, 0):
        return 0.0

    sorted_values = np.sort(cleaned)
    n = sorted_values.size
    index = np.arange(1, n + 1)
    numerator = np.sum((2 * index - n - 1) * sorted_values)
    denominator = n * np.sum(sorted_values)
    return float(numerator / denominator)


def _empty_income_inequality_metrics() -> dict[str, float]:
    return {
        "concentration_index_income_group": float("nan"),
        "absolute_low_high_difference": float("nan"),
        "relative_low_high_ratio": float("nan"),
        "between_group_variance": float("nan"),
        "low_income_mmr_mean": float("nan"),
        "high_income_mmr_mean": float("nan"),
    }


def _validate_income_inequality_metrics(metrics: dict[str, float]) -> None:
    abs_diff = metrics["absolute_low_high_difference"]
    ratio = metrics["relative_low_high_ratio"]
    between_var = metrics["between_group_variance"]
    concentration = metrics["concentration_index_income_group"]

    if np.isfinite(abs_diff) and abs_diff < 0:
        raise ModelingError("Absolute inequality difference must be non-negative")
    if np.isfinite(ratio) and ratio < 0:
        raise ModelingError("Relative inequality ratio must be non-negative")
    if np.isfinite(between_var) and between_var < -1e-12:
        raise ModelingError("Between-group variance must be non-negative")
    if np.isfinite(concentration) and not (-1.0 - 1e-9 <= concentration <= 1.0 + 1e-9):
        raise ModelingError("Concentration index outside expected range [-1, 1]")


def _compute_income_inequality_metrics_for_year(frame: pd.DataFrame) -> dict[str, float]:
    """
    Compute income-stratified inequality metrics for one year.

    Definitions:
    - Concentration index by income group:
      C = (2 / mu) * sum_g(w_g * mu_g * R_g) - 1
      where:
      - g indexes ordered income groups from poorest to richest
      - w_g = n_g / N is the country-share weight in group g
      - mu_g is group-specific mean MMR
      - R_g is the midpoint fractional rank for group g
      - mu = sum_g(w_g * mu_g) is the weighted global mean
    - Absolute low-high difference: |mu_low - mu_high|
    - Relative inequality ratio: mu_low / mu_high
    - Between-group variance:
      sigma_between^2 = sum_g(w_g * (mu_g - mu)^2)

    Mathematical validation:
    - Between-group variance identity check:
      sigma_between^2 = sum_g(w_g * mu_g^2) - mu^2
    """

    if "income_level" not in frame.columns:
        return _empty_income_inequality_metrics()

    grouped_input = frame[["country_iso3", "income_level", "mmr"]].copy()
    grouped_input["mmr"] = pd.to_numeric(grouped_input["mmr"], errors="coerce")
    grouped_input = grouped_input.loc[grouped_input["mmr"].notna() & (grouped_input["mmr"] >= 0)].copy()
    grouped_input["income_group_canonical"] = grouped_input["income_level"].map(_canonical_income_group)
    grouped_input = grouped_input.dropna(subset=["income_group_canonical"])
    if grouped_input.empty:
        return _empty_income_inequality_metrics()

    grouped = (
        grouped_input.groupby("income_group_canonical", as_index=False)
        .agg(
            countries=("country_iso3", "nunique"),
            mmr_mean=("mmr", "mean"),
        )
        .copy()
    )
    grouped["income_rank"] = grouped["income_group_canonical"].map(_CANONICAL_INCOME_RANK)
    grouped = grouped.sort_values("income_rank").reset_index(drop=True)

    total_countries = float(grouped["countries"].sum())
    if total_countries <= 0:
        return _empty_income_inequality_metrics()

    grouped["weight"] = grouped["countries"] / total_countries
    grouped["cumulative_weight"] = grouped["weight"].cumsum()
    grouped["rank_midpoint"] = grouped["cumulative_weight"] - 0.5 * grouped["weight"]

    weighted_mean = float(np.sum(grouped["weight"] * grouped["mmr_mean"]))
    if weighted_mean <= 0 or not np.isfinite(weighted_mean):
        return _empty_income_inequality_metrics()

    concentration_index = float(
        (2.0 / weighted_mean) * np.sum(grouped["weight"] * grouped["mmr_mean"] * grouped["rank_midpoint"])
        - 1.0
    )

    low_row = grouped.loc[grouped["income_group_canonical"] == "low income", "mmr_mean"]
    high_row = grouped.loc[grouped["income_group_canonical"] == "high income", "mmr_mean"]
    low_mean = float(low_row.iloc[0]) if not low_row.empty else float("nan")
    high_mean = float(high_row.iloc[0]) if not high_row.empty else float("nan")

    if np.isfinite(low_mean) and np.isfinite(high_mean):
        absolute_diff = float(abs(low_mean - high_mean))
        relative_ratio = float(low_mean / high_mean) if high_mean > 0 else float("nan")
    else:
        absolute_diff = float("nan")
        relative_ratio = float("nan")

    between_group_variance = float(np.sum(grouped["weight"] * (grouped["mmr_mean"] - weighted_mean) ** 2))
    second_moment_identity = float(np.sum(grouped["weight"] * (grouped["mmr_mean"] ** 2)) - (weighted_mean**2))
    if not np.isclose(between_group_variance, second_moment_identity, rtol=1e-10, atol=1e-10):
        raise ModelingError("Between-group variance identity check failed")

    metrics = {
        "concentration_index_income_group": concentration_index,
        "absolute_low_high_difference": absolute_diff,
        "relative_low_high_ratio": relative_ratio,
        "between_group_variance": max(0.0, between_group_variance),
        "low_income_mmr_mean": low_mean,
        "high_income_mmr_mean": high_mean,
    }
    _validate_income_inequality_metrics(metrics)
    return metrics


def compute_income_inequality_metrics(clean_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly income-stratified inequality metrics for maternal mortality."""

    try:
        required = {"year", "country_iso3", "mmr"}
        missing = required.difference(clean_panel.columns)
        if missing:
            raise ModelingError(f"Missing required columns for income inequality metrics: {sorted(missing)}")

        summaries: list[dict[str, float | int]] = []
        for year, frame in clean_panel.groupby("year"):
            metrics = _compute_income_inequality_metrics_for_year(frame)
            metrics["year"] = int(year)
            summaries.append(metrics)

        return pd.DataFrame(summaries).sort_values("year").reset_index(drop=True)
    except Exception as exc:
        raise ModelingError("Failed to compute income inequality metrics") from exc


def compute_yearly_inequality(clean_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly global and income-stratified maternal mortality inequality metrics."""

    try:
        required = {"year", "country_iso3", "mmr"}
        missing = required.difference(clean_panel.columns)
        if missing:
            raise ModelingError(f"Missing required columns for inequality metrics: {sorted(missing)}")

        def summarize_year(frame: pd.DataFrame) -> pd.Series:
            mmr = frame["mmr"].dropna().to_numpy(dtype=float)
            if mmr.size == 0:
                return pd.Series(
                    {
                        "countries": 0,
                        "mmr_mean": float("nan"),
                        "mmr_median": float("nan"),
                        "mmr_p10": float("nan"),
                        "mmr_p90": float("nan"),
                        "p90_p10_ratio": float("nan"),
                        "gini_mmr": float("nan"),
                        **_empty_income_inequality_metrics(),
                    }
                )

            p10 = float(np.percentile(mmr, 10))
            p90 = float(np.percentile(mmr, 90))
            ratio = float(p90 / p10) if p10 > 0 else float("nan")
            income_metrics = _compute_income_inequality_metrics_for_year(frame)
            return pd.Series(
                {
                    "countries": int(frame["country_iso3"].nunique()),
                    "mmr_mean": float(np.mean(mmr)),
                    "mmr_median": float(np.median(mmr)),
                    "mmr_p10": p10,
                    "mmr_p90": p90,
                    "p90_p10_ratio": ratio,
                    "gini_mmr": _gini(mmr),
                    **income_metrics,
                }
            )

        summaries: list[dict[str, float | int]] = []
        for year, frame in clean_panel.groupby("year"):
            metrics = summarize_year(frame).to_dict()
            metrics["year"] = int(year)
            summaries.append(metrics)

        yearly = pd.DataFrame(summaries).sort_values("year").reset_index(drop=True)
        logger.info("Computed yearly inequality metrics for %s years", len(yearly))
        return yearly
    except Exception as exc:
        raise ModelingError("Failed to compute yearly inequality metrics") from exc


def compute_income_group_summary(clean_panel: pd.DataFrame) -> pd.DataFrame:
    """Compute inequality profile by income group and year."""

    try:
        grouped = (
            clean_panel.groupby(["year", "income_level"], dropna=False)
            .agg(
                countries=("country_iso3", "nunique"),
                mmr_mean=("mmr", "mean"),
                mmr_median=("mmr", "median"),
                gdp_per_capita_mean=("gdp_per_capita", "mean"),
                female_secondary_completion_mean=("female_secondary_completion", "mean"),
            )
            .reset_index()
            .sort_values(["year", "income_level"])
            .reset_index(drop=True)
        )
        logger.info("Computed income group summary with %s rows", len(grouped))
        return grouped
    except Exception as exc:
        raise ModelingError("Failed to compute income group summary") from exc


def build_latest_country_snapshot(clean_panel: pd.DataFrame) -> pd.DataFrame:
    """Extract each country's latest available observation for dashboard snapshot cards."""

    try:
        ordered = clean_panel.sort_values(["country_iso3", "year"]).copy()
        latest_rows = ordered.groupby("country_iso3", as_index=False).tail(1)
        latest_rows = latest_rows.sort_values("country_name").reset_index(drop=True)
        logger.info("Built latest country snapshot with %s countries", len(latest_rows))
        return latest_rows
    except Exception as exc:
        raise ModelingError("Failed to build latest country snapshot") from exc
