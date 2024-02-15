"""Create and save plots for SIMD-Accelerated Distance computations."""

import time

import abd_distances.simd as abd_distances
import scipy.spatial.distance as scipy_distance

from . import utils


def plot_simd_f32() -> None:
    """Plot SIMD speedup for various distance computations."""
    x: list[int] = []
    y_abd: list[float] = []
    y_scipy: list[float] = []

    car = 100
    for d in range(2, 51, 2):
        dim = d * 1000
        data = utils.data_f32(car, dim)

        start = time.perf_counter_ns()
        for a in data:
            for b in data:
                _ = abd_distances.euclidean_f32(a, b)
        end_abd = float(time.perf_counter_ns() - start)
        end_abd = end_abd / (car ** 2) / 1000

        start = time.perf_counter_ns()
        for a in data:
            for b in data:
                _ = scipy_distance.euclidean(a, b)
        end_scipy = float(time.perf_counter_ns() - start)
        end_scipy = end_scipy / (car ** 2) / 1000

        x.append(dim)
        y_abd.append(end_abd)
        y_scipy.append(end_scipy)

    utils.make_plot(x, y_abd, y_scipy, "euclidean", "f32")
