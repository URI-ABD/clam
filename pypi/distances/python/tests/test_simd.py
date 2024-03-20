"""Tests for the SIMD-accelerated distance functions."""

import math
import typing

import numpy
import scipy.spatial.distance as scipy_distance
from abd_distances import simd as simd_distances

Functions = tuple[
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
]

METRICS = ["euclidean", "sqeuclidean", "cosine"]


FUNCTIONS: dict[str, Functions] = {
    "Euclidean": (simd_distances.euclidean, scipy_distance.euclidean),
    "SqEuclidean": (simd_distances.sqeuclidean, scipy_distance.sqeuclidean),
    "Cosine": (simd_distances.cosine, scipy_distance.cosine),
}


def test_simd_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for a in data_f32:
        for b in data_f32:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in FUNCTIONS.items():
                    _simd_helper(a_, b_, name, simd_func, scipy_func, 1e-5)


def test_simd_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for a in data_f64:
        for b in data_f64:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in FUNCTIONS.items():
                    _simd_helper(a_, b_, name, simd_func, scipy_func, 1e-10)


def test_cdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _cdist_helper(
            data_f32,
            data_f32,
            metric,
            simd_distances.cdist,
            1e-5,
        )


def test_cdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _cdist_helper(
            data_f64,
            data_f64,
            metric,
            simd_distances.cdist,
            1e-10,
        )


def test_pdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _pdist_helper(
            data_f32,
            metric,
            simd_distances.pdist,
            1e-5,
        )


def test_pdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _pdist_helper(
            data_f64,
            metric,
            simd_distances.pdist,
            1e-10,
        )


def _simd_helper(  # noqa: PLR0913
    a: numpy.ndarray,
    b: numpy.ndarray,
    metric: str,
    simd_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    scipy_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    dist = simd_func(a, b)
    expected = scipy_func(a, b)
    abs_tol = abs_tol * expected if "cosine" not in metric.lower() else abs_tol
    assert math.isclose(
        dist,
        expected,
        abs_tol=abs_tol,
    ), f"{metric}: Expected: {expected:.8e}, got: {dist:.8e}, rel_tol: {abs_tol:.8e}"


def _cdist_helper(
    a: numpy.ndarray,
    b: numpy.ndarray,
    metric: str,
    simd_func: typing.Callable[[numpy.ndarray, numpy.ndarray, str], numpy.ndarray],
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = simd_func(a, b, metric)
    assert distances.shape == (a.shape[0], b.shape[0])
    expected = scipy_distance.cdist(a, b, metric)
    abs_tol = abs_tol * expected if "cosine" not in metric.lower() else abs_tol
    assert numpy.allclose(
        distances,
        expected,
        atol=abs_tol,
    ), f"{metric}: Expected: {expected}, got: {distances}, rel_tol: {abs_tol}"


def _pdist_helper(
    a: numpy.ndarray,
    metric: str,
    simd_func: typing.Callable[[numpy.ndarray, str], numpy.ndarray],
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = simd_func(a, metric)
    num_distances = (a.shape[0] * (a.shape[0] - 1)) // 2
    assert distances.shape == (num_distances,)
    expected = scipy_distance.pdist(a, metric)
    abs_tol = abs_tol * expected if "cosine" not in metric.lower() else abs_tol
    assert numpy.allclose(
        distances,
        expected,
        atol=abs_tol,
    ), f"{metric}: Expected: {expected}, got: {distances}, rel_tol: {abs_tol}"
