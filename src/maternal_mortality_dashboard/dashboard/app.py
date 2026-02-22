from __future__ import annotations

import logging

from dash import Dash

from maternal_mortality_dashboard.config import Settings, get_settings
from maternal_mortality_dashboard.dashboard.callbacks import register_callbacks
from maternal_mortality_dashboard.dashboard.data_access import load_dashboard_datasets
from maternal_mortality_dashboard.dashboard.layout import build_layout
from maternal_mortality_dashboard.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> Dash:
    resolved_settings = settings or get_settings()
    configure_logging(level=resolved_settings.log_level, log_dir=resolved_settings.log_dir)

    datasets = load_dashboard_datasets(resolved_settings)
    app = Dash(__name__, title="Global Maternal Mortality Inequality Dashboard")
    app.layout = build_layout(
        country_year_df=datasets.country_year,
        latest_snapshot_df=datasets.latest_country_snapshot,
    )
    register_callbacks(app=app, datasets=datasets)
    return app


def run() -> None:
    settings = get_settings()
    app = create_app(settings)
    logger.info("Starting dashboard at %s:%s", settings.dashboard_host, settings.dashboard_port)
    app.run(
        host=settings.dashboard_host,
        port=int(settings.dashboard_port),
        debug=bool(settings.dashboard_debug),
    )


if __name__ == "__main__":
    run()
