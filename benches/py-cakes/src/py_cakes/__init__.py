"""Benchmarking search algorithms for the CAKES paper."""

from .summarize_rust import summarize_rust
from . import utils


def hello(suffix: str) -> str:
    """Return a greeting message."""
    return f"Working on {suffix}..."


__all__ = ["utils", "hello", "summarize_rust"]
