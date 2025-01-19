"""Helpers for the package."""

import logging


def configure_logger(name: str, level: str) -> logging.Logger:
    """Configure a logger with the given name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
