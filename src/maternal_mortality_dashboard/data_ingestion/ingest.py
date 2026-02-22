from __future__ import annotations

import logging

import pandas as pd

from maternal_mortality_dashboard.config import Settings
from maternal_mortality_dashboard.data_ingestion.world_bank_client import WorldBankClient
from maternal_mortality_dashboard.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


def ingest_indicator_panel(settings: Settings) -> pd.DataFrame:
    """Fetch and normalize indicator panel in long format."""

    client = WorldBankClient(settings=settings)
    indicator_map = {
        "mmr": settings.wb_indicator_mmr,
        "gdp_per_capita": settings.wb_indicator_gdp_pc,
        "female_secondary_completion": settings.wb_indicator_female_secondary,
        "female_literacy_rate": settings.wb_indicator_female_literacy,
        "health_expenditure_per_capita": settings.wb_indicator_health_expenditure_pc,
        "skilled_birth_attendance": settings.wb_indicator_skilled_birth_attendance,
        "urban_population_pct": settings.wb_indicator_urban_population_pct,
    }

    frames: list[pd.DataFrame] = []
    try:
        for metric_name, indicator_id in indicator_map.items():
            logger.info(
                "Extracting indicator '%s' (%s) for %s-%s",
                metric_name,
                indicator_id,
                settings.pipeline_start_year,
                settings.pipeline_end_year,
            )
            frame = client.fetch_indicator(
                indicator_id=indicator_id,
                start_year=settings.pipeline_start_year,
                end_year=settings.pipeline_end_year,
            )
            frame = frame[["country_iso3", "country_name", "year", "value"]].copy()
            frame["metric"] = metric_name
            frames.append(frame)
    except Exception as exc:
        raise DataIngestionError("Indicator extraction failed") from exc

    if not frames:
        raise DataIngestionError("No indicator frames were generated")

    indicator_frame = pd.concat(frames, ignore_index=True)
    country_metadata = client.fetch_country_metadata()

    merged = indicator_frame.merge(
        country_metadata,
        on="country_iso3",
        how="left",
        validate="many_to_one",
    )

    merged = merged.loc[
        merged["region"].notna()
        & (merged["region"] != "Aggregates")
        & merged["country_name"].notna()
        & merged["year"].notna()
    ].copy()

    merged["year"] = merged["year"].astype(int)
    merged = merged.sort_values(["country_iso3", "year", "metric"]).reset_index(drop=True)

    logger.info("Extracted panel with %s rows across %s countries", len(merged), merged["country_iso3"].nunique())
    return merged
