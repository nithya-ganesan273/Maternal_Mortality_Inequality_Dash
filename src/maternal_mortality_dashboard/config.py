from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, PositiveInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="global-maternal-mortality-dashboard", alias="APP_NAME")
    app_env: Literal["dev", "staging", "prod", "test"] = Field(default="dev", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    raw_data_dir: Path = Field(default=Path("data/raw"), alias="RAW_DATA_DIR")
    interim_data_dir: Path = Field(default=Path("data/interim"), alias="INTERIM_DATA_DIR")
    processed_data_dir: Path = Field(default=Path("data/processed"), alias="PROCESSED_DATA_DIR")

    world_bank_api_base: HttpUrl = Field(default="https://api.worldbank.org/v2", alias="WORLD_BANK_API_BASE")
    world_bank_per_page: PositiveInt = Field(default=20000, alias="WORLD_BANK_PER_PAGE")
    wb_indicator_mmr: str = Field(default="SH.STA.MMRT", alias="WB_INDICATOR_MMR")
    wb_indicator_gdp_pc: str = Field(default="NY.GDP.PCAP.CD", alias="WB_INDICATOR_GDP_PC")
    wb_indicator_female_secondary: str = Field(
        default="SE.SEC.CUAT.LO.FE.ZS",
        alias="WB_INDICATOR_FEMALE_SECONDARY",
    )
    wb_indicator_female_literacy: str = Field(
        default="SE.ADT.LITR.FE.ZS",
        alias="WB_INDICATOR_FEMALE_LITERACY",
    )
    wb_indicator_health_expenditure_pc: str = Field(
        default="SH.XPD.CHEX.PC.CD",
        alias="WB_INDICATOR_HEALTH_EXPENDITURE_PC",
    )
    wb_indicator_skilled_birth_attendance: str = Field(
        default="SH.STA.BRTC.ZS",
        alias="WB_INDICATOR_SKILLED_BIRTH_ATTENDANCE",
    )
    wb_indicator_urban_population_pct: str = Field(
        default="SP.URB.TOTL.IN.ZS",
        alias="WB_INDICATOR_URBAN_POPULATION_PCT",
    )

    pipeline_start_year: int = Field(default=2000, alias="PIPELINE_START_YEAR")
    pipeline_end_year: int = Field(default=2023, alias="PIPELINE_END_YEAR")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")
    request_timeout_seconds: PositiveInt = Field(default=30, alias="REQUEST_TIMEOUT_SECONDS")
    request_retries: PositiveInt = Field(default=3, alias="REQUEST_RETRIES")
    regression_min_observations: PositiveInt = Field(default=150, alias="REGRESSION_MIN_OBSERVATIONS")
    regression_optional_predictor_min_coverage: float = Field(
        default=0.65,
        alias="REGRESSION_OPTIONAL_PREDICTOR_MIN_COVERAGE",
    )

    dashboard_host: str = Field(default="0.0.0.0", alias="DASH_HOST")
    dashboard_port: int = Field(default=8050, alias="DASH_PORT")
    dashboard_debug: bool = Field(default=False, alias="DASH_DEBUG")

    @model_validator(mode="after")
    def validate_pipeline_window(self) -> "Settings":
        if self.pipeline_start_year > self.pipeline_end_year:
            raise ValueError("PIPELINE_START_YEAR must be less than or equal to PIPELINE_END_YEAR")
        if not 0.0 <= self.regression_optional_predictor_min_coverage <= 1.0:
            raise ValueError("REGRESSION_OPTIONAL_PREDICTOR_MIN_COVERAGE must be between 0 and 1")
        return self

    def ensure_runtime_directories(self) -> None:
        for path in (
            self.log_dir,
            self.data_dir,
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_directories()
    return settings
