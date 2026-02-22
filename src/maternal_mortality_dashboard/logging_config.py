from __future__ import annotations

import logging.config
from pathlib import Path


def configure_logging(level: str, log_dir: Path) -> None:
    """Configure structured console + rotating file logging."""

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "application.log"

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": level,
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": level,
                    "formatter": "standard",
                    "filename": str(log_path),
                    "maxBytes": 10_000_000,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": level,
                "handlers": ["console", "file"],
            },
        }
    )
