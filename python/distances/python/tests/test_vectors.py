"""Tests for the distances for vectors."""

import math
import typing

import numpy
import scipy.spatial.distance as scipy_distance
from abd_distances import vectors as vector_distances

Functions = tuple[
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
]


def scipy_l3(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Compute the L3 norm using scipy."""
    return scipy_distance.minkowski(a, b, 3)


def scipy_l4(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Compute the L4 norm using scipy."""
    return scipy_distance.minkowski(a, b, 4)


VECTOR_F32: dict[str, Functions] = {
    "Chebyshev f32": (vector_distances.chebyshev_f32, scipy_distance.chebyshev),
    "Euclidean f32": (vector_distances.euclidean_f32, scipy_distance.euclidean),
    "SqEuclidean f32": (vector_distances.euclidean_sq_f32, scipy_distance.sqeuclidean),
    "L3 f32": (vector_distances.l3_distance_f32, scipy_l3),
    "L4 f32": (vector_distances.l4_distance_f32, scipy_l4),
    "Manhattan f32": (vector_distances.manhattan_f32, scipy_distance.cityblock),
    "Canberra f32": (vector_distances.canberra_f32, scipy_distance.canberra),
    "Cosine f32": (vector_distances.cosine_f32, scipy_distance.cosine),
}

VECTOR_F64: dict[str, Functions] = {
    "Chebyshev f64": (vector_distances.chebyshev_f64, scipy_distance.chebyshev),
    "Euclidean f64": (vector_distances.euclidean_f64, scipy_distance.euclidean),
    "SqEuclidean f64": (vector_distances.euclidean_sq_f64, scipy_distance.sqeuclidean),
    "L3 f64": (vector_distances.l3_distance_f64, scipy_l3),
    "L4 f64": (vector_distances.l4_distance_f64, scipy_l4),
    "Manhattan f64": (vector_distances.manhattan_f64, scipy_distance.cityblock),
    "Canberra f64": (vector_distances.canberra_f64, scipy_distance.canberra),
    "Cosine f64": (vector_distances.cosine_f64, scipy_distance.cosine),
}


def test_vector_f32(data_f32: numpy.ndarray):
    """Test the vector distance functions."""
    for a in data_f32:
        for b in data_f32:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in VECTOR_F32.items():
                    _vector_helper(a_, b_, name, simd_func, scipy_func, 1e-5)


def test_vector_f64(data_f64: numpy.ndarray):
    """Test the vector distance functions."""
    for a in data_f64:
        for b in data_f64:
            for a_, b_ in [(a, a), (a, b), (b, a), (b, b)]:
                for name, (simd_func, scipy_func) in VECTOR_F64.items():
                    _vector_helper(a_, b_, name, simd_func, scipy_func, 1e-10)


def _vector_helper(  # noqa: PLR0913
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


def test_bray_curtis_u32(data_u32: numpy.ndarray):
    """Test the `bray_curtis_u32` function."""
    a, b = data_u32[0], data_u32[1]
    dist = vector_distances.bray_curtis_u32(a, b)
    expected = scipy_distance.braycurtis(a, b)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_bray_curtis_u64(data_u64: numpy.ndarray):
    """Test the `bray_curtis_u64` function."""
    a, b = data_u64[0], data_u64[1]
    dist = vector_distances.bray_curtis_u64(a, b)
    expected = scipy_distance.braycurtis(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_hamming_i32(data_i32: numpy.ndarray):
    """Test the `hamming_i32` function."""
    a, b = data_i32[0], data_i32[1]
    dist = vector_distances.hamming_i32(a, b)
    dist = dist / a.size
    expected = scipy_distance.hamming(a, b)
    rel_tol = 1e-4
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_hamming_i64(data_i64: numpy.ndarray):
    """Test the `hamming_i64` function."""
    a, b = data_i64[0], data_i64[1]
    dist = vector_distances.hamming_i64(a, b)
    dist = dist / a.size
    expected = scipy_distance.hamming(a, b)
    rel_tol = 1e-8
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_cdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("chebyshev", "chebyshev"),
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("manhattan", "cityblock"),
        ("canberra", "canberra"),
        ("cosine", "cosine"),
    ]:
        _cdist_helper(
            data_f32,
            data_f32,
            abd_name,
            vector_distances.cdist_f32,
            scipy_name,
            1e-4,
        )


def test_cdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("chebyshev", "chebyshev"),
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("manhattan", "cityblock"),
        ("canberra", "canberra"),
        ("cosine", "cosine"),
    ]:
        _cdist_helper(
            data_f64,
            data_f64,
            abd_name,
            vector_distances.cdist_f64,
            scipy_name,
            1e-8,
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
    dist = simd_func(a, b, abd_name)
    assert dist.shape == (a.shape[0], b.shape[0])
    expected = scipy_distance.cdist(a, b, scipy_name)
    abs_tol = abs_tol * expected if "cosine" not in abd_name.lower() else abs_tol
    assert numpy.allclose(
        dist,
        expected,
        rtol=abs_tol,
    ), f"{abd_name}: Expected: {expected}, got: {dist}, rel_tol: {abs_tol}"


def test_pdist_f32(data_f32: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("chebyshev", "chebyshev"),
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("manhattan", "cityblock"),
        ("canberra", "canberra"),
        ("cosine", "cosine"),
    ]:
        _pdist_helper(
            data_f32,
            abd_name,
            vector_distances.pdist_f32,
            scipy_name,
            1e-4,
        )


def test_pdist_f64(data_f64: numpy.ndarray):
    """Test the SIMD-accelerated distance functions."""
    for abd_name, scipy_name in [
        ("chebyshev", "chebyshev"),
        ("euclidean", "euclidean"),
        ("euclidean_sq", "sqeuclidean"),
        ("manhattan", "cityblock"),
        ("canberra", "canberra"),
        ("cosine", "cosine"),
    ]:
        _pdist_helper(
            data_f64,
            abd_name,
            vector_distances.pdist_f64,
            scipy_name,
            1e-8,
        )


def _pdist_helper(
    a: numpy.ndarray,
    abd_name: str,
    simd_func: typing.Callable[[numpy.ndarray, str], numpy.ndarray],
    scipy_name: str,
    abs_tol: float,
) -> None:
    """Helper function for the SIMD-accelerated distance functions."""
    dist = simd_func(a, abd_name)
    expected = scipy_distance.pdist(a, scipy_name)
    abs_tol = abs_tol * expected if "cosine" not in abd_name.lower() else abs_tol
    assert numpy.allclose(
        dist,
        expected,
        rtol=abs_tol,
    ), f"{abd_name}: Expected: {expected}, got: {dist}, rel_tol: {abs_tol}"
