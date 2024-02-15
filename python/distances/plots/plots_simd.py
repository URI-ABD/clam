"""Create and save plots for SIMD-Accelerated Distance computations."""

import time
import typing

import abd_distances.simd as abd_distances
import numpy
import scipy.spatial.distance as scipy_distance
import tqdm

from . import utils

Functions = tuple[
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[int, int], numpy.ndarray],
]


FUNCTION_PAIRS: dict[str, Functions] = {
    "Euclidean f32": (abd_distances.euclidean_f32, scipy_distance.euclidean, utils.data_f32),  # noqa: E501
    "Euclidean f64": (abd_distances.euclidean_f64, scipy_distance.euclidean, utils.data_f64),  # noqa: E501
    "SquaredEuclidean f32": (abd_distances.euclidean_sq_f32, scipy_distance.sqeuclidean, utils.data_f32),  # noqa: E501
    "SquaredEuclidean f64": (abd_distances.euclidean_sq_f64, scipy_distance.sqeuclidean, utils.data_f64),  # noqa: E501
    "Cosine f32": (abd_distances.cosine_f32, scipy_distance.cosine, utils.data_f32),  # noqa: E501
    "Cosine f64": (abd_distances.cosine_f64, scipy_distance.cosine, utils.data_f64),  # noqa: E501
}


def make_plots() -> None:
    """Plot SIMD speedup for various distance computations."""
    for name, (abd_func, scipy_func, gen_data) in tqdm.tqdm(FUNCTION_PAIRS.items()):
        [dist_name, dtype] = name.split(" ")

        x: list[int] = []
        y_abd: list[float] = []
        y_scipy: list[float] = []

        car = 100
        for d in range(2, 51, 2):
            dim = d * 100
            data = gen_data(car, dim)

            start = time.perf_counter_ns()
            for a in data:
                for b in data:
                    _ = abd_func(a, b)
            end_abd = float(time.perf_counter_ns() - start)
            end_abd = end_abd / (car ** 2) / 1000

            start = time.perf_counter_ns()
            for a in data:
                for b in data:
                    _ = scipy_func(a, b)
            end_scipy = float(time.perf_counter_ns() - start)
            end_scipy = end_scipy / (car ** 2) / 1000

            x.append(dim)
            y_abd.append(end_abd)
            y_scipy.append(end_scipy)

        utils.make_plot(x, y_abd, y_scipy, dist_name, dtype)
