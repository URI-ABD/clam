"""Create and save plots for SIMD-Accelerated Distance computations."""

import time
import typing

import abd_distances.simd as simd_distances
import abd_distances.vectors as vector_distances
import numpy
import scipy.spatial.distance as scipy_distance
import tqdm

from . import utils

Functions = tuple[
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[numpy.ndarray, numpy.ndarray], float],
    typing.Callable[[int, int], numpy.ndarray],
]


def scipy_l3(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Compute the L3 norm using scipy."""
    return scipy_distance.minkowski(a, b, 3)


def scipy_l4(a: numpy.ndarray, b: numpy.ndarray) -> float:
    """Compute the L4 norm using scipy."""
    return scipy_distance.minkowski(a, b, 4)


FUNCTION_PAIRS: dict[str, Functions] = {
    "SIMD Euclidean, f32": (simd_distances.euclidean_f32, scipy_distance.euclidean, utils.data_f32),  # noqa: E501
    "SIMD Euclidean, f64": (simd_distances.euclidean_f64, scipy_distance.euclidean, utils.data_f64),  # noqa: E501
    "SIMD Squared Euclidean, f32": (simd_distances.sqeuclidean_f32, scipy_distance.sqeuclidean, utils.data_f32),  # noqa: E501
    "SIMD Squared Euclidean, f64": (simd_distances.sqeuclidean_f64, scipy_distance.sqeuclidean, utils.data_f64),  # noqa: E501
    "SIMD Cosine, f32": (simd_distances.cosine_f32, scipy_distance.cosine, utils.data_f32),  # noqa: E501
    "SIMD Cosine, f64": (simd_distances.cosine_f64, scipy_distance.cosine, utils.data_f64),  # noqa: E501
    "Chebyshev, f32": (vector_distances.chebyshev_f32, scipy_distance.chebyshev, utils.data_f32),  # noqa: E501
    "Chebyshev, f64": (vector_distances.chebyshev_f64, scipy_distance.chebyshev, utils.data_f64),  # noqa: E501
    "Euclidean, f32": (vector_distances.euclidean_f32, scipy_distance.euclidean, utils.data_f32),  # noqa: E501
    "Euclidean, f64": (vector_distances.euclidean_f64, scipy_distance.euclidean, utils.data_f64),  # noqa: E501
    "Squared Euclidean, f32": (vector_distances.sqeuclidean_f32, scipy_distance.sqeuclidean, utils.data_f32),  # noqa: E501
    "Squared Euclidean, f64": (vector_distances.sqeuclidean_f64, scipy_distance.sqeuclidean, utils.data_f64),  # noqa: E501
    "L3, f32": (vector_distances.l3_distance_f32, scipy_l3, utils.data_f32),  # noqa: E501
    "L3, f64": (vector_distances.l3_distance_f64, scipy_l3, utils.data_f64),  # noqa: E501
    "L4, f32": (vector_distances.l4_distance_f32, scipy_l4, utils.data_f32),  # noqa: E501
    "L4, f64": (vector_distances.l4_distance_f64, scipy_l4, utils.data_f64),  # noqa: E501
    "Manhattan, f32": (vector_distances.manhattan_f32, scipy_distance.cityblock, utils.data_f32),  # noqa: E501
    "Manhattan, f64": (vector_distances.manhattan_f64, scipy_distance.cityblock, utils.data_f64),  # noqa: E501
    "Canberra, f32": (vector_distances.canberra_f32, scipy_distance.canberra, utils.data_f32),  # noqa: E501
    "Canberra, f64": (vector_distances.canberra_f64, scipy_distance.canberra, utils.data_f64),  # noqa: E501
    "Cosine, f32": (vector_distances.cosine_f32, scipy_distance.cosine, utils.data_f32),  # noqa: E501
    "Cosine, f64": (vector_distances.cosine_f64, scipy_distance.cosine, utils.data_f64),  # noqa: E501
    "Bray-Curtis, u32": (vector_distances.braycurtis_u32, scipy_distance.braycurtis, utils.data_u32),  # noqa: E501
    "Bray-Curtis, u64": (vector_distances.braycurtis_u64, scipy_distance.braycurtis, utils.data_u64),  # noqa: E501
    "Hamming, i32": (vector_distances.hamming_i32, scipy_distance.hamming, utils.data_i32),  # noqa: E501
    "Hamming, i64": (vector_distances.hamming_i64, scipy_distance.hamming, utils.data_i64),  # noqa: E501
}


def make_plots() -> None:
    """Plot SIMD speedup for various distance computations."""
    for name, (abd_func, scipy_func, gen_data) in tqdm.tqdm(
        FUNCTION_PAIRS.items(),
        desc="Vector Distances",
    ):
        [dist_name, dtype] = name.split(", ")
        dist_name = "-".join(dist_name.split(" "))

        x: list[int] = []
        y_abd: list[float] = []
        y_scipy: list[float] = []

        car = 100
        for d in range(5, 51, 5):
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

        utils.make_plot(
            x=x,
            y_abd=y_abd,
            y_comp=y_scipy,
            fn_name=dist_name,
            dt_name=dtype,
        )
