"""Distances and similarity measures for vectors."""

import typing

import numpy

from .abd_distances import typeless_vectors as abd_vectors


def chebyshev(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Chebyshev distance between two vectors."""
    return abd_vectors.chebyshev(a, b)


def euclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return abd_vectors.euclidean(a, b)


def sqeuclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Squared Euclidean distance between two vectors."""
    return abd_vectors.sqeuclidean(a, b)


def manhattan(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Manhattan distance between two vectors."""
    return abd_vectors.manhattan(a, b)


def minkowski(a: numpy.ndarray, b: numpy.ndarray, p: float) -> float:
    """Minkowski distance between two vectors."""
    return abd_vectors.minkowski(a, b, p)


def cosine(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return abd_vectors.cosine(a, b)


def cdist(
    a: numpy.ndarray,
    b: numpy.ndarray,
    metric: str,
    p: typing.Optional[int] = None,
) -> numpy.ndarray:
    """Compute distance between each pair of the two collections of inputs."""
    return abd_vectors.cdist(a, b, metric, p)


def pdist(
    a: numpy.ndarray,
    metric: str,
    p: typing.Optional[int] = None,
) -> numpy.ndarray:
    """Pairwise distances between observations in n-dimensional space."""
    return abd_vectors.pdist(a, metric, p)


__all__ = [
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "minkowski",
    "cosine",
    "cdist",
    "pdist",
]
