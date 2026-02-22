from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maternal_mortality_dashboard.config import Settings
from maternal_mortality_dashboard.data_cleaning.transform import clean_indicator_panel
from maternal_mortality_dashboard.data_ingestion.ingest import ingest_indicator_panel
from maternal_mortality_dashboard.exceptions import PipelineExecutionError
from maternal_mortality_dashboard.io import write_json, write_parquet
from maternal_mortality_dashboard.modeling.inequality_metrics import (
    build_latest_country_snapshot,
    compute_income_group_summary,
    compute_yearly_inequality,
)
from maternal_mortality_dashboard.modeling.ecological_regression import run_ecological_regression

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineArtifacts:
    raw_indicator_path: Path
    interim_clean_panel_path: Path
    dashboard_country_year_path: Path
    dashboard_yearly_inequality_path: Path
    dashboard_income_group_summary_path: Path
    dashboard_latest_snapshot_path: Path
    ecological_regression_dataset_path: Path
    ecological_regression_coefficients_path: Path
    ecological_regression_model_summary_csv_path: Path
    ecological_regression_model_summary_text_path: Path
    ecological_regression_vif_path: Path
    ecological_regression_residuals_vs_fitted_plot_path: Path
    ecological_regression_qq_plot_path: Path
    metadata_path: Path


def _signature(df: pd.DataFrame) -> str:
    ordered = df.sort_values(list(df.columns)).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(ordered, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _build_metadata(
    settings: Settings,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    income_df: pd.DataFrame,
    ecological_regression_details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pipeline_window": {
            "start_year": settings.pipeline_start_year,
            "end_year": settings.pipeline_end_year,
        },
        "source_indicators": {
            "mmr": settings.wb_indicator_mmr,
            "gdp_per_capita": settings.wb_indicator_gdp_pc,
            "female_secondary_completion": settings.wb_indicator_female_secondary,
            "female_literacy_rate": settings.wb_indicator_female_literacy,
            "health_expenditure_per_capita": settings.wb_indicator_health_expenditure_pc,
            "skilled_birth_attendance": settings.wb_indicator_skilled_birth_attendance,
            "urban_population_pct": settings.wb_indicator_urban_population_pct,
        },
        "row_counts": {
            "raw_indicator_panel": int(len(raw_df)),
            "clean_country_year_panel": int(len(clean_df)),
            "yearly_inequality": int(len(yearly_df)),
            "income_group_summary": int(len(income_df)),
        },
        "country_coverage": {
            "raw_unique_countries": int(raw_df["country_iso3"].nunique()),
            "clean_unique_countries": int(clean_df["country_iso3"].nunique()),
        },
        "ecological_regression": ecological_regression_details,
        "input_signature_sha256": _signature(raw_df),
    }


def run_pipeline(settings: Settings) -> PipelineArtifacts:
    """Run ingestion -> cleaning -> modeling and persist dashboard artifacts."""

    np.random.seed(settings.random_seed)
    logger.info(
        "Starting pipeline for years %s-%s",
        settings.pipeline_start_year,
        settings.pipeline_end_year,
    )

    artifacts = PipelineArtifacts(
        raw_indicator_path=settings.raw_data_dir
        / f"world_bank_indicators_{settings.pipeline_start_year}_{settings.pipeline_end_year}.parquet",
        interim_clean_panel_path=settings.interim_data_dir
        / f"clean_panel_{settings.pipeline_start_year}_{settings.pipeline_end_year}.parquet",
        dashboard_country_year_path=settings.processed_data_dir / "dashboard_country_year.parquet",
        dashboard_yearly_inequality_path=settings.processed_data_dir / "dashboard_yearly_inequality.parquet",
        dashboard_income_group_summary_path=settings.processed_data_dir
        / "dashboard_income_group_summary.parquet",
        dashboard_latest_snapshot_path=settings.processed_data_dir
        / "dashboard_latest_country_snapshot.parquet",
        ecological_regression_dataset_path=settings.processed_data_dir / "ecological_regression_dataset.parquet",
        ecological_regression_coefficients_path=settings.processed_data_dir / "ecological_regression_coefficients.csv",
        ecological_regression_model_summary_csv_path=settings.processed_data_dir
        / "ecological_regression_model_summary.csv",
        ecological_regression_model_summary_text_path=settings.processed_data_dir
        / "ecological_regression_model_summary.txt",
        ecological_regression_vif_path=settings.processed_data_dir / "ecological_regression_vif.csv",
        ecological_regression_residuals_vs_fitted_plot_path=settings.processed_data_dir
        / "ecological_regression_residuals_vs_fitted.png",
        ecological_regression_qq_plot_path=settings.processed_data_dir
        / "ecological_regression_residuals_qq_plot.png",
        metadata_path=settings.processed_data_dir / "pipeline_metadata.json",
    )

    try:
        raw_df = ingest_indicator_panel(settings=settings)
        write_parquet(raw_df, artifacts.raw_indicator_path)

        clean_df = clean_indicator_panel(raw_df).sort_values(["country_iso3", "year"]).reset_index(drop=True)
        write_parquet(clean_df, artifacts.interim_clean_panel_path)
        write_parquet(clean_df, artifacts.dashboard_country_year_path)

        yearly_df = compute_yearly_inequality(clean_df)
        income_df = compute_income_group_summary(clean_df)
        latest_snapshot_df = build_latest_country_snapshot(clean_df)
        ecological_regression_result = run_ecological_regression(
            clean_panel=clean_df,
            output_dir=settings.processed_data_dir,
            minimum_observations=int(settings.regression_min_observations),
            optional_predictor_min_coverage=float(settings.regression_optional_predictor_min_coverage),
        )

        write_parquet(yearly_df, artifacts.dashboard_yearly_inequality_path)
        write_parquet(income_df, artifacts.dashboard_income_group_summary_path)
        write_parquet(latest_snapshot_df, artifacts.dashboard_latest_snapshot_path)

        metadata = _build_metadata(
            settings,
            raw_df,
            clean_df,
            yearly_df,
            income_df,
            ecological_regression_details={
                "n_observations": ecological_regression_result.n_observations,
                "predictors_used": ecological_regression_result.predictors_used,
                "r_squared": ecological_regression_result.r_squared,
                "adj_r_squared": ecological_regression_result.adj_r_squared,
                "covariance_estimator": "HC3",
                "coefficients_path": str(ecological_regression_result.artifacts.coefficients_path),
                "model_summary_csv_path": str(ecological_regression_result.artifacts.model_summary_csv_path),
                "model_summary_text_path": str(ecological_regression_result.artifacts.model_summary_text_path),
                "vif_path": str(ecological_regression_result.artifacts.vif_path),
                "residuals_vs_fitted_plot_path": str(
                    ecological_regression_result.artifacts.residuals_vs_fitted_plot_path
                ),
                "qq_plot_path": str(ecological_regression_result.artifacts.qq_plot_path),
            },
        )
        write_json(metadata, artifacts.metadata_path)
    except Exception as exc:
        raise PipelineExecutionError("Pipeline execution failed") from exc

    logger.info("Pipeline completed successfully")
    return artifacts


def serialize_artifacts(artifacts: PipelineArtifacts) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(artifacts).items()}
