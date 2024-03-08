"""Tests for the SIMD-accelerated distance functions."""

import math
import typing
from functools import partial

import numpy
import scipy.spatial.distance as scipy_distance
from abd_distances import vectors as abd_distances

Functions = tuple[
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
]


FUNCTIONS: dict[str, Functions] = {
    "BrayCurtis": (abd_distances.braycurtis, scipy_distance.braycurtis),
    "Canberra": (abd_distances.canberra, scipy_distance.canberra),
    "Chebyshev": (abd_distances.chebyshev, scipy_distance.chebyshev),
    "Euclidean": (abd_distances.euclidean, scipy_distance.euclidean),
    "SqEuclidean": (abd_distances.sqeuclidean, scipy_distance.sqeuclidean),
    "Manhattan": (abd_distances.manhattan, scipy_distance.cityblock),
    "Cosine": (abd_distances.cosine, scipy_distance.cosine),
}


METRICS = [
    "braycurtis",
    "canberra",
    "chebyshev",
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "cosine",
]


def test_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for a in data_f32:
        for b in data_f32:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in FUNCTIONS.items():
                    _check_distances(a_, b_, name, simd_func, scipy_func, 1e-5)


def test_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for a in data_f64:
        for b in data_f64:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in FUNCTIONS.items():
                    _check_distances(a_, b_, name, simd_func, scipy_func, 1e-10)


def test_minkowski_f32(data_f32: numpy.ndarray):
    """Test the Minkowski distance function."""
    for p in range(3, 7):
        abd_mink = partial(abd_distances.minkowski, p=p)
        scipy_mink = partial(scipy_distance.minkowski, p=p)
        for a in data_f32:
            for b in data_f32:
                for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                    _check_distances(a_, b_, f"Minkowski, p={p}", abd_mink, scipy_mink, 1e-5)


def test_minkowski_f64(data_f64: numpy.ndarray):
    """Test the Minkowski distance function."""
    for p in range(3, 7):
        abd_mink = partial(abd_distances.minkowski, p=p)
        scipy_mink = partial(scipy_distance.minkowski, p=p)
        for a in data_f64:
            for b in data_f64:
                for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                    _check_distances(a_, b_, f"Minkowski, p={p}", abd_mink, scipy_mink, 1e-10)


def test_cdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _cdist_helper(
            data_f32,
            data_f32,
            metric,
            1e-5,
        )


def test_cdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _cdist_helper(
            data_f64,
            data_f64,
            metric,
            1e-10,
        )


def test_pdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _pdist_helper(
            data_f32,
            metric,
            1e-5,
        )


def test_pdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for metric in METRICS:
        _pdist_helper(
            data_f64,
            metric,
            1e-10,
        )


def _check_distances(  # noqa: PLR0913
    a: numpy.ndarray,
    b: numpy.ndarray,
    name: str,
    abd_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    scipy_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    rel_tol: float,
) -> None:
    """Helper function for running tests."""
    dist = abd_func(a, b)
    expected = scipy_func(a, b)
    if "cosine" in name.lower():
        # The `scipy` implementation of the cosine distance has a lot of numerical
        # instability. So we use a small absolute tolerance instead of a relative one.
        assert math.isclose(
            dist,
            expected,
            abs_tol=1e-4,
        ), f"{name}: Expected: {expected:.8e}, got: {dist:.8e}, abs_tol: {1e-4:.8e}"
    else:
        assert math.isclose(
            dist,
            expected,
            rel_tol=rel_tol,
        ), f"{name}: Expected: {expected:.8e}, got: {dist:.8e}, rel_tol: {rel_tol:.8e}"


def _cdist_helper(
    a: numpy.ndarray,
    b: numpy.ndarray,
    metric: str,
    rel_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = abd_distances.cdist(a, b, metric)
    assert distances.shape == (a.shape[0], b.shape[0])
    expected = scipy_distance.cdist(a, b, metric)
    if "cosine" in metric.lower():
        # The `scipy` implementation of the cosine distance has a lot of numerical
        # instability. So we use a small absolute tolerance instead of a relative one.
        assert numpy.allclose(
            distances,
            expected,
            atol=1e-4,
        ), f"{metric}: Expected: {expected}, got: {distances}, abs_tol: {1e-4:.8e}"
    else:
        assert numpy.allclose(
            distances,
            expected,
            rtol=rel_tol,
        ), f"{metric}: Expected: {expected}, got: {distances}, rel_tol: {rel_tol:.8e}"


def _pdist_helper(
    a: numpy.ndarray,
    metric: str,
    rel_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = abd_distances.pdist(a, metric)
    num_distances = (a.shape[0] * (a.shape[0] - 1)) // 2
    assert distances.shape == (num_distances,)
    expected = scipy_distance.pdist(a, metric)
    if "cosine" in metric.lower():
        # The `scipy` implementation of the cosine distance has a lot of numerical
        # instability. So we use a small absolute tolerance instead of a relative one.
        assert numpy.allclose(
            distances,
            expected,
            atol=1e-4,
        ), f"{metric}: Expected: {expected}, got: {distances}, abs_tol: {1e-4:.8e}"
    else:
        assert numpy.allclose(
            distances,
            expected,
            rtol=rel_tol,
        ), f"{metric}: Expected: {expected}, got: {distances}, rel_tol: {rel_tol:.8e}"
