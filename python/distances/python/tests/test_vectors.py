"""Tests for the distances for vectors."""

import math

import abd_distances
import numpy
import scipy.spatial.distance as scipy_distance


def test_chebyshev_f32(data_f32: numpy.ndarray):
    """Test the `chebyshev_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.chebyshev_f32(a, b)
    expected = scipy_distance.chebyshev(a, b)
    assert dist == expected, f"Expected: {expected}, got: {dist}"


def test_chebyshev_f64(data_f64: numpy.ndarray):
    """Test the `chebyshev_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.chebyshev_f64(a, b)
    expected = scipy_distance.chebyshev(a, b)
    assert dist == expected, f"Expected: {expected}, got: {dist}"


def test_euclidean_f32(data_f32: numpy.ndarray):
    """Test the `euclidean_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.euclidean_f32(a, b)
    expected = scipy_distance.euclidean(a, b)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_euclidean_f64(data_f64: numpy.ndarray):
    """Test the `euclidean_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.euclidean_f64(a, b)
    expected = scipy_distance.euclidean(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_euclidean_sq_f32(data_f32: numpy.ndarray):
    """Test the `euclidean_sq_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.euclidean_sq_f32(a, b)
    expected = scipy_distance.sqeuclidean(a, b)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_euclidean_sq_f64(data_f64: numpy.ndarray):
    """Test the `euclidean_sq_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.euclidean_sq_f64(a, b)
    expected = scipy_distance.sqeuclidean(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_l3_distance_f32(data_f32: numpy.ndarray):
    """Test the `l3_distance_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.l3_distance_f32(a, b)
    expected = scipy_distance.minkowski(a, b, 3)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_l3_distance_f64(data_f64: numpy.ndarray):
    """Test the `l3_distance_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.l3_distance_f64(a, b)
    expected = scipy_distance.minkowski(a, b, 3)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_l4_distance_f32(data_f32: numpy.ndarray):
    """Test the `l4_distance_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.l4_distance_f32(a, b)
    expected = scipy_distance.minkowski(a, b, 4)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_l4_distance_f64(data_f64: numpy.ndarray):
    """Test the `l4_distance_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.l4_distance_f64(a, b)
    expected = scipy_distance.minkowski(a, b, 4)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_manhattan_f32(data_f32: numpy.ndarray):
    """Test the `manhattan_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.manhattan_f32(a, b)
    expected = scipy_distance.cityblock(a, b)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_manhattan_f64(data_f64: numpy.ndarray):
    """Test the `manhattan_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.manhattan_f64(a, b)
    expected = scipy_distance.cityblock(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_bray_curtis_u32(data_u32: numpy.ndarray):
    """Test the `bray_curtis_u32` function."""
    a, b = data_u32[0], data_u32[1]
    dist = abd_distances.vectors.bray_curtis_u32(a, b)
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
    dist = abd_distances.vectors.bray_curtis_u64(a, b)
    expected = scipy_distance.braycurtis(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_canberra_f32(data_f32: numpy.ndarray):
    """Test the `canberra_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.canberra_f32(a, b)
    expected = scipy_distance.canberra(a, b)
    rel_tol = 1e-6 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_canberra_f64(data_f64: numpy.ndarray):
    """Test the `canberra_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.canberra_f64(a, b)
    expected = scipy_distance.canberra(a, b)
    rel_tol = 1e-12 * expected
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_cosine_f32(data_f32: numpy.ndarray):
    """Test the `cosine_f32` function."""
    a, b = data_f32[0], data_f32[1]
    dist = abd_distances.vectors.cosine_f32(a, b)
    expected = scipy_distance.cosine(a, b)
    rel_tol = 1e-4
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_cosine_f64(data_f64: numpy.ndarray):
    """Test the `cosine_f64` function."""
    a, b = data_f64[0], data_f64[1]
    dist = abd_distances.vectors.cosine_f64(a, b)
    expected = scipy_distance.cosine(a, b)
    rel_tol = 1e-8
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"


def test_hamming_i32(data_i32: numpy.ndarray):
    """Test the `hamming_i32` function."""
    a, b = data_i32[0], data_i32[1]
    dist = abd_distances.vectors.hamming_i32(a, b)
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
    dist = abd_distances.vectors.hamming_i64(a, b)
    dist = dist / a.size
    expected = scipy_distance.hamming(a, b)
    rel_tol = 1e-8
    assert math.isclose(
        dist,
        expected,
        rel_tol=rel_tol,
    ), f"Expected: {expected}, got: {dist}"
