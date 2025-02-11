"""Plots for the dimension reduction from CLAM-MBED."""

from .plot_dim_red import plot as plot_dim_red
from . import utils


def hello(suffix: str) -> str:
    """Return a greeting message."""
    return f"Working on {suffix}..."


__all__ = ["utils", "hello", "plot_dim_red"]
