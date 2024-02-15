"""Tests for the distances for vectors."""

import math

import abd_distances


def test_euclidean_f32():
    a = [1, 2, 3]
    b = [4, 5, 6]

    differences = tuple(a_ - b_ for a_, b_ in zip(a, b))
    expected_dist = math.hypot(*differences)
    dist = abd_distances.vectors.euclidean_f32(a, b)
    diff = abs(dist - expected_dist)
    assert diff < 1e-6, f"Expected {expected_dist}, got {dist}"
