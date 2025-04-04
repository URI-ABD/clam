"""Plotting dimensionality reduction using UMAP."""

import pathlib

import numpy
from matplotlib import pyplot as plt


def plot(
    inp_path: pathlib.Path,
    labels_path: pathlib.Path,
    ax: list[plt.Axes],
) -> None:
    """Plot dimensionality reduction results using UMAP."""

    # Load the data
    embedding = numpy.load(inp_path)
    if labels_path.exists():
        labels = numpy.load(labels_path)
    else:
        labels = numpy.zeros(embedding.shape[0], dtype=bool)

    # colors will be blue for 0 and red for 1
    colors = ["red" if label else "blue" for label in labels]

    # Plot the UMAP results
    ax[0].scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("UMAP X-Y")

    ax[1].scatter(
        embedding[:, 0],
        embedding[:, 2],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Z")
    ax[1].set_title("UMAP X-Z")

    ax[2].scatter(
        embedding[:, 1],
        embedding[:, 2],
        s=10,
        alpha=0.5,
        c=colors,
    )
    ax[2].set_xlabel("Y")
    ax[2].set_ylabel("Z")
    ax[2].set_title("UMAP Y-Z")
