"""Create and save plots for SIMD-Accelerated Distance computations."""

import time
import typing

import abd_distances.strings as string_distances
import tqdm

from . import utils

FUNCTIONS: dict[str, typing.Callable[[str, str], int]] = {
    "Hamming": string_distances.hamming,
    "Levenshtein": string_distances.levenshtein,
    "Needleman-Wunsch": string_distances.needleman_wunsch,
}


def make_plots() -> None:
    """Plot speed of string distance computations."""
    alphabet = "ACTGN"

    for name, func in tqdm.tqdm(FUNCTIONS.items(), desc="String distances"):
        x: list[int] = []
        y_abd: list[float] = []

        car = 10
        str_lengths = list(range(1, 9)) + list(range(10, 51, 5))
        str_lengths = [i * 100 for i in str_lengths]
        for dim in str_lengths:
            data = utils.gen_strings(car, dim, alphabet)

            start = time.perf_counter()
            for a in data:
                for b in data:
                    _ = func(a, b)
            end = float(time.perf_counter() - start) * 1000
            if "hamming" in name.lower():
                end *= 1000
            end = end / (car ** 2)

            x.append(dim)
            y_abd.append(end)

        y_units = "µs" if "hamming" in name.lower() else "ms"
        utils.make_plot(
            x=x,
            y_abd=y_abd,
            fn_name=name,
            dt_name="str",
            y_units=y_units,
            x_label="String Length",
        )
