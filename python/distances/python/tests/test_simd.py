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


SIMD_F32: dict[str, Functions] = {
    "Euclidean f32": (simd_distances.euclidean_f32, scipy_distance.euclidean),
    "SqEuclidean f32": (simd_distances.euclidean_sq_f32, scipy_distance.sqeuclidean),
    "Cosine f32": (simd_distances.cosine_f32, scipy_distance.cosine),
}

SIMD_F64: dict[str, Functions] = {
    "Euclidean f64": (simd_distances.euclidean_f64, scipy_distance.euclidean),
    "SqEuclidean f64": (simd_distances.euclidean_sq_f64, scipy_distance.sqeuclidean),
    "Cosine f64": (simd_distances.cosine_f64, scipy_distance.cosine),
}


def test_simd_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    a, b = data_f32[0], data_f32[1]
    for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
        for name, (simd_func, scipy_func) in SIMD_F32.items():
            _simd_helper(a_, b_, name, simd_func, scipy_func, 1e-5)


def test_simd_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    a, b = data_f64[0], data_f64[1]
    for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
        for name, (simd_func, scipy_func) in SIMD_F64.items():
            _simd_helper(a_, b_, name, simd_func, scipy_func, 1e-10)


def _simd_helper(  # noqa: PLR0913
    a: numpy.ndarray,
    b: numpy.ndarray,
    name: str,
    simd_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    scipy_func: typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    dist = simd_func(a, b)
    expected = scipy_func(a, b)
    abs_tol = abs_tol * expected if "cosine" not in name.lower() else abs_tol
    assert math.isclose(
        dist,
        expected,
        abs_tol=abs_tol,
    ), f"{name}: Expected: {expected:.8e}, got: {dist:.8e}, rel_tol: {abs_tol:.8e}"


def test_cdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("cosine", "cosine"),
    ]:
        _cdist_helper(
            data_f32,
            data_f32,
            abd_name,
            simd_distances.cdist_f32,
            scipy_name,
            1e-5,
        )


def test_cdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("cosine", "cosine"),
    ]:
        _cdist_helper(
            data_f64,
            data_f64,
            abd_name,
            simd_distances.cdist_f64,
            scipy_name,
            1e-10,
        )


def _cdist_helper(  # noqa: PLR0913
    a: numpy.ndarray,
    b: numpy.ndarray,
    abd_name: str,
    simd_func: typing.Callable[[numpy.ndarray, numpy.ndarray, str], numpy.ndarray],
    scipy_name: str,
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = simd_func(a, b, abd_name)
    assert distances.shape == (a.shape[0], b.shape[0])
    expected = scipy_distance.cdist(a, b, scipy_name)
    abs_tol = abs_tol * expected if "cosine" not in abd_name.lower() else abs_tol
    assert numpy.allclose(
        distances,
        expected,
        atol=abs_tol,
    ), f"{abd_name}: Expected: {expected}, got: {distances}, rel_tol: {abs_tol}"


def test_pdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("cosine", "cosine"),
    ]:
        _pdist_helper(
            data_f32,
            abd_name,
            simd_distances.pdist_f32,
            scipy_name,
            1e-5,
        )


def test_pdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("cosine", "cosine"),
    ]:
        _pdist_helper(
            data_f64,
            abd_name,
            simd_distances.pdist_f64,
            scipy_name,
            1e-10,
        )


def _pdist_helper(
    a: numpy.ndarray,
    abd_name: str,
    simd_func: typing.Callable[[numpy.ndarray, str], numpy.ndarray],
    scipy_name: str,
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    distances = simd_func(a, abd_name)
    num_distances = (a.shape[0] * (a.shape[0] - 1)) // 2
    assert distances.shape == (num_distances,)
    expected = scipy_distance.pdist(a, scipy_name)
    abs_tol = abs_tol * expected if "cosine" not in abd_name.lower() else abs_tol
    assert numpy.allclose(
        distances,
        expected,
        atol=abs_tol,
    ), f"{abd_name}: Expected: {expected}, got: {distances}, rel_tol: {abs_tol}"
