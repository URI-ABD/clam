"""Plotting dimensionality reduction using UMAP."""

import pathlib

import numpy
import umap
from matplotlib import pyplot as plt


def plot(
    inp_path: pathlib.Path,
    ax: plt.Axes,
) -> None:
    """Plot dimensionality reduction results using UMAP."""

    # Read the input data as a numpy array
    raw_data = numpy.load(inp_path)

    # Perform UMAP dimensionality reduction
    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=2,
    )

    # Fit the UMAP model
    embedding = reducer.fit_transform(raw_data)

    # Plot the UMAP results
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=10,
        alpha=0.5,
    )
    ax.set_title("UMAP")
