"""Reading the tables from Rust and making plots."""

import pathlib

import matplotlib.pyplot as plt
import numpy
import pandas

BALL_TYPES = {
    "depth": numpy.uint32,
    "cardinality": numpy.uint64,
    "radius": numpy.uint32,
    "lfd": numpy.float32,
    "arg_center": numpy.uint64,
    "arg_radial": numpy.uint64,
}

SQUISHY_BALL_TYPES = {
    **BALL_TYPES,
    "offset": numpy.uint64,
    "unitary_cost": numpy.uint32,
    "recursive_cost": numpy.uint32,
}


def draw_plots(
    pre_trim_path: pathlib.Path,
    post_trim_path: pathlib.Path,
) -> None:
    """Read the ball table from the csv file."""
    # Read the pre-trim and post-trim dataframes from the csv files.
    pre_trim_df = pandas.read_csv(pre_trim_path, dtype=SQUISHY_BALL_TYPES)
    post_trim_df = pandas.read_csv(post_trim_path, dtype=SQUISHY_BALL_TYPES)

    # Drop all rows where "recursive_cost" is not positive.
    pre_trim_df = pre_trim_df[pre_trim_df["recursive_cost"] > 0]
    post_trim_df = post_trim_df[post_trim_df["recursive_cost"] > 0]

    # Drop all rows where "cardinality" is too small.
    min_cardinality = 1
    pre_trim_df = pre_trim_df[pre_trim_df["cardinality"] >= min_cardinality]
    post_trim_df = post_trim_df[post_trim_df["cardinality"] >= min_cardinality]

    # Create a new column called "ratio" that is the ratio of "recursive_cost" to "unitary_cost"
    pre_trim_df["ratio"] = pre_trim_df["recursive_cost"] / pre_trim_df["unitary_cost"]
    post_trim_df["ratio"] = post_trim_df["recursive_cost"] / post_trim_df["unitary_cost"]

    # Calculate the maximum ratio of "recursive_cost" to "unitary_cost" in the
    # pre-trim and post-trim dataframes.
    max_ratio = numpy.ceil(max(pre_trim_df["ratio"].max(), post_trim_df["ratio"].max()))

    # Calculate the maximum "lfd" in the pre-trim and post-trim dataframes.
    max_lfd = max(pre_trim_df["lfd"].max(), post_trim_df["lfd"].max())

    dfs = {
        "pre_trim": pre_trim_df,
        "post_trim": post_trim_df,
    }

    for name, df in dfs.items():
        # Make a scatter plot with "depth" on the x-axis, "lfd" on the y-axis,
        # and "ratio" as the color.
        ax = df.plot.scatter(
            x="depth",
            y="lfd",
            s=0.2,
            c="ratio",
            cmap="bwr",
            vmin=0,
            vmax=numpy.ceil(max_ratio),
        )
        # Set the minimum and maximum values of the y-axis.
        ax.set_ylim(0, numpy.ceil(max_lfd))

        # Set the title of the plot to be the name of the dataframe.
        title = f"Recursive / Unitary ratio for {name} Clusters"
        ax.set_title(title)

        # Save the plot to a file with the name of the dataframe.
        plot_path = pre_trim_path.parent / f"{name}-ratio.png"
        plt.savefig(plot_path, dpi=300)

        # Make a scatter plot with "depth" on the x-axis, "ratio" on the y-axis,
        # and "lfd" as the color.
        ax = df.plot.scatter(
            x="depth",
            y="ratio",
            s=0.2,
            c="lfd",
            cmap="bwr",
            vmin=0,
            vmax=numpy.ceil(max_lfd),
        )
        # Set the minimum and maximum values of the y-axis.
        ax.set_ylim(0, numpy.ceil(max_ratio))

        # Set the title of the plot to be the name of the dataframe.
        ax.set_title(title)

        # Save the plot to a file with the name of the dataframe.
        plot_path = pre_trim_path.parent / f"{name}-lfd.png"
        plt.savefig(plot_path, dpi=300)

        # Close the plots.
        plt.close("all")


def normalized_color_scale(values: numpy.ndarray) -> numpy.ndarray:
    """Apply Gaussian normalization to the values and return the result."""
    # Calculate the mean and standard deviation of the values.
    mean = values.mean()
    std = values.std()

    # Apply Gaussian normalization to the values.
    return (values - mean) / std
