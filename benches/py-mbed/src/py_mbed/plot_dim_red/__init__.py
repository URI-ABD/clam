"""Plotting the results of dimensionality reduction."""

import pathlib

from matplotlib import pyplot as plt

from . import plot_mbed
from . import plot_umap


def plot(
    original_path: pathlib.Path,
    mbed_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """Plot dimensionality reduction results."""

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))

    plot_umap.plot(original_path, ax[0])
    plot_mbed.plot(mbed_path, ax[1])

    plt.tight_layout()
    plt.savefig(out_dir / "dim_red_results.png", dpi=300)
    plt.close()


__all__ = ["plot"]
