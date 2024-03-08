"""Type stubs for vector distances."""

import numpy

def braycurtis(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Bray-Curtis distance between two vectors."""

def canberra(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Canberra distance between two vectors."""

def chebyshev(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Chebyshev distance between two vectors."""

def euclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Euclidean distance between two vectors."""

def sqeuclidean(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Squared Euclidean distance between two vectors."""

def manhattan(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Manhattan distance between two vectors."""

def minkowski(a: numpy.ndarray, b: numpy.ndarray, p: float) -> float:
    """Minkowski distance between two vectors."""

def cosine(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Cosine similarity between two vectors."""

def cdist(
    a: numpy.ndarray,
    b: numpy.ndarray,
    metric: str,
    p: int | None = None,
) -> numpy.ndarray:
    """Compute distance between each pair of the two collections of inputs.

    `p` is only used for Minkowski distance.
    """

def pdist(
    a: numpy.ndarray,
    metric: str,
    p: int | None = None,
) -> numpy.ndarray:
    """Pairwise distances between observations in n-dimensional space.

    `p` is only used for Minkowski distance.
    """
