"""Structured logging setup using loguru.

In production (LOG_FORMAT=json), every record is serialised as a single JSON line.
In development (LOG_FORMAT=text), a human-readable coloured format is used.
"""

import sys
from typing import Any

from loguru import logger

from app.utils.config import get_settings


def _json_sink(message: Any) -> None:
    """Write pre-serialised loguru JSON records to stdout."""
    print(message, end="", flush=True)


def setup_logging() -> None:
    """Configure loguru based on application settings. Call once at startup."""
    settings = get_settings()
    logger.remove()

    if settings.log_format == "json":
        logger.add(
            _json_sink,
            level=settings.log_level,
            serialize=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        )
        logger.add(
            sys.stderr,
            format=fmt,
            level=settings.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

    logger.info(
        "Logging initialised",
        level=settings.log_level,
        format=settings.log_format,
    )


def get_logger(name: str) -> Any:
    """Return a child logger bound to *name*."""
    return logger.bind(module=name)
