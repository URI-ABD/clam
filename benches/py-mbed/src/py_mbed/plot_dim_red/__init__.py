"""Plotting the results of dimensionality reduction."""

import pathlib

from matplotlib import pyplot as plt

from . import gif_mbed
from . import plot_mbed
from . import plot_umap


def plot(
    original_path: pathlib.Path,
    mbed_dir: pathlib.Path,
    mbed_name: str,
    labels_path: pathlib.Path,
    out_dir: pathlib.Path,
) -> None:
    """Plot dimensionality reduction results."""

    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(1, 2, figsize=(16, 10))

    mbed_path = mbed_dir / f"{mbed_name}-reduced.npy"
    plot_umap.plot(original_path, labels_path, ax[0])
    plot_mbed.plot(mbed_path, labels_path, ax[1])

    plt.tight_layout()
    plt.savefig(out_dir / "dim_red_results.png", dpi=300)
    plt.close()

    gif_mbed.plot(mbed_dir, mbed_name, out_dir)


__all__ = ["plot"]
