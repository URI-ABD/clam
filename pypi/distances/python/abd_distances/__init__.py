"""A package for computing distances."""

from . import simd
from . import strings
from . import vectors

setattr(vectors, "cityblock", vectors.manhattan)  # noqa: B010

__all__ = [
    "simd",
    "strings",
    "vectors",
]

__version__ = "1.0.2"
