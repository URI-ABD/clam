"""Distances and similarity measures for vectors."""

from .abd_distances import vectors as abd_vectors

euclidean_f32 = abd_vectors.euclidean_f32


__all__ = [
    "euclidean_f32",
]
