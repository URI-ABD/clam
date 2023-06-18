import pathlib
import typing

import reports
from matplotlib import pyplot


def _violin(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:
    ax.violinplot(
        dataset=[[c.lfd for c in clusters] for clusters in clusters_by_depth],
        positions=list(range(len(clusters_by_depth))),
    )
    return ax


def _heat(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
):
    raise NotImplementedError


def plot_lfd_vs_depth(
    mode: typing.Literal["violin", "heat"],
    tree: reports.TreeReport,
    clusters_by_depth: list[list[reports.ClusterReport]],
    show: bool,
    output_dir: pathlib.Path,
):
    figure: pyplot.Figure = pyplot.figure(figsize=(16, 10), dpi=300)
    title = ", ".join(
        [
            f"name = {tree.data_name}",
            f"metric = {tree.metric_name}",
            f"cardinality = {tree.cardinality}",
            f"dimensionality = {tree.dimensionality}",
            f"build_time = {tree.build_time:3.2e} (sec)",
        ],
    )
    figure.suptitle(title)

    ax = (_violin if mode == "violin" else _heat)(
        clusters_by_depth,
        pyplot.axes((0.05, 0.1, 0.9, 0.85)),
    )

    ax.set_xlabel("depth - num_clusters")
    ax.set_ylabel("local fractal dimension")
    ax.set_xticks(
        list(range(len(clusters_by_depth))),
        [f"{d}-{len(clusters)}" for d, clusters in enumerate(clusters_by_depth)],
        rotation=270,
    )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(f"{mode}-{tree.data_name}__{tree.metric_name}.png"),
            dpi=300,
        )
    pyplot.close(figure)
