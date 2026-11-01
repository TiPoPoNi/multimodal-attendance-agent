"""Logging configuration for Multimodal Attendance & Engagement Agent.

Provides structured, clean logging to console and local file without
persisting sensitive personal data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "attendance_agent",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Set up and configure a named logger.

    Args:
        name: Name of the logger.
        log_level: Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional path to write log output.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple duplicate handlers if already configured
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional File Handler
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Could not initialize log file %s: %s", log_file, exc)

    logger.propagate = False
    return logger


def get_logger(name: str = "attendance_agent") -> logging.Logger:
    """Retrieve or create a logger with standard settings.

    Args:
        name: Subsystem or module name.

    Returns:
        logging.Logger: The requested logger.
    """
    return logging.getLogger(name)

