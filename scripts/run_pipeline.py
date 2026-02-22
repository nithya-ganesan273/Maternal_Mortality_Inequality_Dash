from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from maternal_mortality_dashboard.config import get_settings
from maternal_mortality_dashboard.logging_config import configure_logging
from maternal_mortality_dashboard.pipeline.orchestrator import run_pipeline, serialize_artifacts


def main() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, log_dir=settings.log_dir)

    logger = logging.getLogger(__name__)
    artifacts = run_pipeline(settings=settings)
    logger.info("Pipeline artifacts: %s", serialize_artifacts(artifacts))


if __name__ == "__main__":
    main()
