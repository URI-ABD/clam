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

    # Calculate the maximum values of some columns.
    max_ratio = numpy.ceil(max(pre_trim_df["ratio"].max(), post_trim_df["ratio"].max()))  # noqa: F841
    max_lfd = max(pre_trim_df["lfd"].max(), post_trim_df["lfd"].max())
    max_depth = max(pre_trim_df["depth"].max(), post_trim_df["depth"].max())
    max_radius = max(pre_trim_df["radius"].max(), post_trim_df["radius"].max())  # noqa: F841

    dfs = {
        "pre_trim": pre_trim_df,
        "post_trim": post_trim_df,
    }

    cmap = "cool"
    for name, df in dfs.items():
        # Make a scatter plot with "depth" on the x-axis, "ratio" on the y-axis,
        # and "lfd" as the color.
        df["color"], mean, std = normalized_color_scale(df["lfd"])
        ax = df.plot.scatter(
            x="depth",
            y="lfd",
            s=0.2,
            c="ratio",
            cmap=cmap,
            vmin=0,
            vmax=numpy.ceil(max_lfd),
        )
        # Set the minimum and maximum values of the y-axis.
        ax.set_xlim(0, max_depth)
        ax.set_ylim(1, numpy.ceil(max_lfd))

        # Set the title of the plot to be the name of the dataframe.
        title = f"Recursive / Unitary ratio for {pre_trim_path.stem}"
        ax.set_title(title)

        # Save the plot to a file with the name of the dataframe.
        plot_path = pre_trim_path.parent / f"{pre_trim_path.stem}-{name}-ratio.png"
        plt.savefig(plot_path, dpi=300)

        # Close the plots.
        plt.close("all")


def normalized_color_scale(values: numpy.ndarray) -> tuple[numpy.ndarray, float, float]:
    """Apply Gaussian normalization to the values and return the result."""
    # Calculate the mean and standard deviation of the values.
    mean = values.mean()
    std = values.std()

    # Apply Gaussian normalization to the values.
    return (values - mean) / std, mean, std
