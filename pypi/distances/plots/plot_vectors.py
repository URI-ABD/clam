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


FUNCTION_PAIRS: dict[str, Functions] = {
    "SIMD Euclidean, f32": (
        simd_distances.euclidean,
        scipy_distance.euclidean,
        utils.data_f32,
    ),
    "SIMD Euclidean, f64": (
        simd_distances.euclidean,
        scipy_distance.euclidean,
        utils.data_f64,
    ),
    "SIMD Squared Euclidean, f32": (
        simd_distances.sqeuclidean,
        scipy_distance.sqeuclidean,
        utils.data_f32,
    ),
    "SIMD Squared Euclidean, f64": (
        simd_distances.sqeuclidean,
        scipy_distance.sqeuclidean,
        utils.data_f64,
    ),
    "SIMD Cosine, f32": (simd_distances.cosine, scipy_distance.cosine, utils.data_f32),
    "SIMD Cosine, f64": (simd_distances.cosine, scipy_distance.cosine, utils.data_f64),
    "Chebyshev, f32": (
        vector_distances.chebyshev,
        scipy_distance.chebyshev,
        utils.data_f32,
    ),
    "Chebyshev, f64": (
        vector_distances.chebyshev,
        scipy_distance.chebyshev,
        utils.data_f64,
    ),
    "Euclidean, f32": (
        vector_distances.euclidean,
        scipy_distance.euclidean,
        utils.data_f32,
    ),
    "Euclidean, f64": (
        vector_distances.euclidean,
        scipy_distance.euclidean,
        utils.data_f64,
    ),
    "Squared Euclidean, f32": (
        vector_distances.sqeuclidean,
        scipy_distance.sqeuclidean,
        utils.data_f32,
    ),
    "Squared Euclidean, f64": (
        vector_distances.sqeuclidean,
        scipy_distance.sqeuclidean,
        utils.data_f64,
    ),
    "Manhattan, f32": (
        vector_distances.manhattan,
        scipy_distance.cityblock,
        utils.data_f32,
    ),
    "Manhattan, f64": (
        vector_distances.manhattan,
        scipy_distance.cityblock,
        utils.data_f64,
    ),
    "Canberra, f32": (
        vector_distances.canberra,
        scipy_distance.canberra,
        utils.data_f32,
    ),  # type: ignore[attr-defined]
    "Canberra, f64": (
        vector_distances.canberra,
        scipy_distance.canberra,
        utils.data_f64,
    ),  # type: ignore[attr-defined]
    "Cosine, f32": (vector_distances.cosine, scipy_distance.cosine, utils.data_f32),
    "Cosine, f64": (vector_distances.cosine, scipy_distance.cosine, utils.data_f64),
    "Bray-Curtis, u32": (
        vector_distances.braycurtis,
        scipy_distance.braycurtis,
        utils.data_u32,
    ),  # type: ignore[attr-defined]
    "Bray-Curtis, u64": (
        vector_distances.braycurtis,
        scipy_distance.braycurtis,
        utils.data_u64,
    ),  # type: ignore[attr-defined]
    # "Hamming, i32": (vector_distances.hamming, scipy_distance.hamming, utils.data_i32),  # type: ignore[attr-defined]  # noqa: E501
    # "Hamming, i64": (vector_distances.hamming, scipy_distance.hamming, utils.data_i64),  # type: ignore[attr-defined]  # noqa: E501
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
            end_abd = end_abd / (car**2) / 1000

            start = time.perf_counter_ns()
            for a in data:
                for b in data:
                    _ = scipy_func(a, b)
            end_scipy = float(time.perf_counter_ns() - start)
            end_scipy = end_scipy / (car**2) / 1000

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
