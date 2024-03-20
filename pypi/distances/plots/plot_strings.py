"""Create and save plots for SIMD-Accelerated Distance computations."""

from __future__ import annotations

import time
import typing

import abd_distances.strings as string_distances
import editdistance
import tqdm

from . import utils

Functions = tuple[
    typing.Callable[[str, str], int],
    tuple[typing.Optional[str], typing.Optional[typing.Callable[[str, str], int]]],
]

FUNCTIONS: dict[str, Functions] = {
    "Hamming": (string_distances.hamming, (None, None)),
    "Levenshtein": (string_distances.levenshtein, ("roy-ht", editdistance.eval)),
    "Needleman-Wunsch": (string_distances.needleman_wunsch, (None, None)),
}


def make_plots() -> None:  # noqa: C901
    """Plot speed of string distance computations."""
    alphabet = "ACTGN"

    for name, (abd_func, (comp_name, comp_func)) in tqdm.tqdm(
        FUNCTIONS.items(),
        desc="String distances",
    ):
        x: list[int] = []
        y_abd: list[float] = []

        y_comp: list[float] | None = None
        if comp_name:
            y_comp = []

        car = 10
        str_lengths = list(range(1, 9)) + list(range(10, 51, 5))
        str_lengths = [i * 100 for i in str_lengths]
        for dim in str_lengths:
            x.append(dim)

            data = utils.gen_strings(car, dim, alphabet)

            start = time.perf_counter()
            for a in data:
                for b in data:
                    _ = abd_func(a, b)
            end = float(time.perf_counter() - start) * 1000
            if "hamming" in name.lower():
                end *= 1000
            end = end / (car**2)
            y_abd.append(end)

            if y_comp is not None:
                start = time.perf_counter()
                for a in data:
                    for b in data:
                        _ = comp_func(a, b)
                end_comp = float(time.perf_counter() - start) * 1000
                if "hamming" in name.lower():
                    end_comp *= 1000
                end_comp = end_comp / (car**2)
                y_comp.append(end_comp)

        y_units = "µs" if "hamming" in name.lower() else "ms"
        y_comp_label = comp_name if y_comp else ""

        utils.make_plot(
            x=x,
            y_abd=y_abd,
            fn_name=name,
            dt_name="str",
            y_units=y_units,
            x_label="String Length",
            y_comp=y_comp,
            y_comp_label=y_comp_label,
        )
