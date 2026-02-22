from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from maternal_mortality_dashboard.config import Settings
from maternal_mortality_dashboard.exceptions import DashboardDataError, PipelineIOError
from maternal_mortality_dashboard.io import read_parquet


@dataclass(frozen=True)
class DashboardDatasets:
    country_year: pd.DataFrame
    yearly_inequality: pd.DataFrame
    income_group_summary: pd.DataFrame
    latest_country_snapshot: pd.DataFrame
    regression_coefficients: pd.DataFrame
    regression_model_summary: pd.DataFrame


def _required_file(path: Path) -> Path:
    if not path.exists():
        raise DashboardDataError(
            f"Required dashboard artifact missing: {path}. Run scripts/run_pipeline.py first."
        )
    return path


def load_dashboard_datasets(settings: Settings) -> DashboardDatasets:
    try:
        country_year = read_parquet(
            _required_file(settings.processed_data_dir / "dashboard_country_year.parquet")
        )
        yearly_ineq = read_parquet(
            _required_file(settings.processed_data_dir / "dashboard_yearly_inequality.parquet")
        )
        income_summary = read_parquet(
            _required_file(settings.processed_data_dir / "dashboard_income_group_summary.parquet")
        )
        latest_snapshot = read_parquet(
            _required_file(settings.processed_data_dir / "dashboard_latest_country_snapshot.parquet")
        )
    except PipelineIOError as exc:
        raise DashboardDataError("Failed reading dashboard artifacts") from exc
    except Exception as exc:
        raise DashboardDataError("Failed loading dashboard parquet artifacts") from exc

    try:
        regression_coefficients = pd.read_csv(
            _required_file(settings.processed_data_dir / "ecological_regression_coefficients.csv")
        )
        regression_model_summary = pd.read_csv(
            _required_file(settings.processed_data_dir / "ecological_regression_model_summary.csv")
        )
    except Exception as exc:
        raise DashboardDataError("Failed loading ecological regression artifacts") from exc

    return DashboardDatasets(
        country_year=country_year,
        yearly_inequality=yearly_ineq,
        income_group_summary=income_summary,
        latest_country_snapshot=latest_snapshot,
        regression_coefficients=regression_coefficients,
        regression_model_summary=regression_model_summary,
    )
