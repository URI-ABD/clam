"""SIMD accelerated distance functions."""

from .abd_distances import simd

euclidean = simd.euclidean
sqeuclidean = simd.sqeuclidean
cosine = simd.cosine
cdist = simd.cdist
pdist = simd.pdist


__all__ = [
    "euclidean",
    "sqeuclidean",
    "cosine",
    "cdist",
    "pdist",
]
