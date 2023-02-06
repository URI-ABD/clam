import pathlib
import typing
import seaborn
import pandas
import math
import numpy 
from matplotlib import pyplot


import reports


#Attempt at refactoring this module to be a little easier to read. The attempt is not going well. 


def insert_empty_bins(bins: numpy.ndarray, grouped_data: pandas.Series):
    values = list(grouped_data.values)
    for i in range(len(bins)):
        if round(bins[i], 3) not in [key.left for key in grouped_data.keys()]:
            values.insert(i, 0)
    return values



def _violin_parent_child_ratio(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    pc_ratio_index: int,
):
    ax.violinplot(
        dataset=[
            [c.ratios[pc_ratio_index] for c in clusters] for clusters in clusters_by_depth
        ],
        positions=list(range(len(clusters_by_depth))),
    )

    return ax



def _violin(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    dependent_variable: typing.Literal[
        "old_radius", "new_radius", "radii_ratio", "lfd"
    ],
    x_label: str, 
    clamping_function: typing.Literal[
        "None", "radii_comparison", "max"
    ] = "None",
):
    if dependent_variable == "radii_ratio": 
        dataset = [ [c.radii_ratio for c in clusters if c.radii_ratio != 1.0] for clusters in clusters_by_depth],

    elif dependent_variable == "new_radius": 
        dataset = [[c.radius for c in clusters] for clusters in clusters_by_depth]

    elif dependent_variable == "old_radius": 
        dataset = [[c.old_radius for c in clusters] for clusters in clusters_by_depth] 
    
    elif dependent_variable == "lfd": 
        dataset = [[c.lfd for c in clusters] for clusters in clusters_by_depth]

        
    ax.violinplot(
        dataset,
        positions=list(range(len(clusters_by_depth))),
    )

    _set_violin_labels(clusters_by_depth, ax, x_label, dependent_variable, dataset, clamping_function)

    return ax

def _line_radius_comparison(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:

    ax.plot(
        list(range(len(clusters_by_depth))), 
        [numpy.mean([c.radius for c in clusters]) for clusters in clusters_by_depth],
        label = "new radius",
    )

    ax.plot(
        list(range(len(clusters_by_depth))), 
        [numpy.mean([c.old_radius for c in clusters]) for clusters in clusters_by_depth],
        label = "old radius",
    )   

    return ax


def _line(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    dependent_variable: typing.Literal[
        "radii_ratio", "lfd"
    ],
    x_label: str, 
    clamping_function: typing.Literal[
        "None", "radii_comparison", "max"
    ] = "None",)  -> pyplot.Axes:

    if dependent_variable == "radii_ratio": 
        dataset = [numpy.mean([c.radii_ratio for c in clusters]) for clusters in clusters_by_depth]
    
    elif dependent_variable == "lfd": 
        dataset = [numpy.mean([c.lfd for c in clusters]) for clusters in clusters_by_depth]

    ax.plot(list(range(len(clusters_by_depth))), 
        dataset,
    )

    _set_line_labels(clusters_by_depth, ax, x_label, dependent_variable, dataset, clamping_function)

    return ax


def _scatter_radii_ratio_by_cardinality(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
) -> pyplot.Axes:
    ax.scatter(
        x=[c.cardinality for clusters in clusters_by_depth for c in clusters][10:],
        y = [c.radii_ratio for clusters in clusters_by_depth for c in clusters][10:]
    )
    return ax


def _heat_lfd(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    normalization_type: typing.Literal["cluster", "cardinality"] = "cardinality",
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
) -> pyplot.Axes:

    # Regroups clusters_by_depth based on desired size of depth buckets and
    # replaes each cluster with its lfd s.t. the ith sublist in bucketed_lfds_by_depth
    # contains lfds of all clusters whose depth is between i*depth_bucket_step
    # and (i+1)*depth_bucket_step
    bucketed_lfds_by_depth = [
        [
            cluster.lfd
            for depth in clusters_by_depth
            for cluster in depth
            if clusters_by_depth.index(depth) // depth_bucket_step == d
        ]
        for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
    ]

    if normalization_type == "cardinality":
        bucketed_cardinalities_by_depth = [
            [
                cluster.cardinality
                for depth in clusters_by_depth
                for cluster in depth
                if clusters_by_depth.index(depth) // depth_bucket_step == d
            ]
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ]

    # Creates list of bins for lfd based on desired number of lfd buckets and the max
    # lfd across all clusters in the tree
    max_lfd = max([lfd for depth in bucketed_lfds_by_depth for lfd in depth])
    bins = numpy.arange(0, max_lfd + max_lfd / num_lfd_buckets, max_lfd / num_lfd_buckets)

    # .cut() segments each sublist in bucketed_lfds_by_depth by lfd bucket and value_counts()
    # counts how many clusters within a particular depth sublist fall into each lfd category
    # Thse lists need to be reversed with [::-1] to correct the fact that seaborn likes to invert
    # the y axes in heatmaps.
    # The array also needs to be transposed in order to get lfd on the y axis.
    if normalization_type == "cluster":
        data = numpy.transpose(
            [
                (pandas.cut(lfds_set, bins=bins).value_counts() / len(lfds_set))[
                    ::-1
                ].to_list()
                for lfds_set in bucketed_lfds_by_depth
            ]
        )
    else:
        data = numpy.transpose(
            [
                insert_empty_bins(
                    bins,
                    pandas.Series(cards_set, index=pandas.cut(lfds_set, bins).to_list())
                    .groupby(level=0)
                    .sum()
                    / sum(cards_set),
                )[::-1]
                for lfds_set, cards_set in zip(
                    bucketed_lfds_by_depth, bucketed_cardinalities_by_depth
                )
            ]
        )

    # vmax currently clamped to 0.5. May cause issues with datasets whose lfd is more consistent
    # across clusters in the same depth. Setting to 1 makes it difficult to see color variation; allowing
    # it to be automatic makes it difficult to compare heatmaps for different datasets.
    return seaborn.heatmap(
        data=data,
        vmin=0,
        vmax=0.5,
        cmap="Reds",
        annot=False,
        linewidths=0.75,
        linecolor="white",
        ax=ax,
    )


def _set_violin_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    x_label: str, 
    y_label: str, 
    dataset: list[list[float]],
    clamping_function: typing.Literal[
        "None", "radii_comparison", "max"
    ] = "None",
): 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks(
        [d for d in range(len(dataset))],
        [f"{d}" for d in range(len(dataset))],
        rotation=270,
    )

    if clamping_function == "radii_comparison": 
        _clamp_radii_comparison(clusters_by_depth, ax)
    elif clamping_function == "max": 
        _clamp_to_max(dataset, ax)

    return


def _set_line_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    x_label: str, 
    y_label: str, 
    dataset: list[list[float]],
    clamping_function: typing.Literal[
        "None", "radii_comparison", "max"
    ] = "None",
): 
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks(
        [d for d in range(len(dataset))],
        [f"{d}" for d in range(len(dataset))],
        rotation=270,
    )

    if clamping_function == "radii_comparison": 
        _clamp_radii_comparison(ax, dataset)
    elif clamping_function == "max": 
        _clamp_to_max(ax, dataset)

    return


def _clamp_radii_comparison(clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes, 
    buffer: int = 1):

    max_radius = max(clusters_by_depth[0][0].old_radius, clusters_by_depth[0][0].radius)
    high_value = int(math.ceil(max_radius / 1000.0)) * 1000 

    step = int(high_value / 10) 
    ax.set_yticks(
        [d for d in range(0, high_value + buffer, step)],
        [f"{d}" for d in  range(0, high_value + buffer, step)],
    )
    return

def _clamp_to_max(dataset:  list[list[float]],
    ax: pyplot.Axes,
    buffer: int = 1, 
    num_buckets: int = 10):

    high_value = max([datum for depth in dataset for datum in depth]) 
    step = int(max / num_buckets)

    ax.set_yticks(
        [d for d in range(0, high_value + buffer, step)],
        [f"{d}" for d in  range(0, high_value + buffer, step)],
    )
    
    return


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
        ]
    )
    figure.suptitle(title)

    if mode == "violin": 
        ax = _violin(clusters_by_depth,
        pyplot.axes((0.05, 0.1, 0.9, 0.85)),
        "lfd")
    else: 
        ax = _heat_lfd(clusters_by_depth,
        pyplot.axes((0.05, 0.1, 0.9, 0.85)),)

    # ax = (_violin_lfd if mode == "violin" else _heat_lfd)(
    #     clusters_by_depth,
    #     pyplot.axes((0.05, 0.1, 0.9, 0.85)),
    # )

    if mode == "violin":
        _set_violin_labels(clusters_by_depth, ax, "depth", "lfd")
    else:
        _set_heat_lfd_labels( clusters_by_depth, ax)

    # (_set_violin_lfd_labels if mode == "violin" else _set_heat_lfd_labels)(
    #     clusters_by_depth, ax
    # )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(f"{mode}-lfd-{tree.data_name}__{tree.metric_name}.png"),
            dpi=300,
        )
    pyplot.close(figure)

    return







def _heat_parent_child_ratio(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_index: int,
    normalization_type: typing.Literal["cluster", "cardinality"] = "cardinality",
    num_ratio_buckets: int = 50,
    depth_bucket_step: int = 2,
):

    bucketed_ratios_by_depth = [
        [
            cluster.ratios[ratio_index]
            for depth in clusters_by_depth
            for cluster in depth
            if clusters_by_depth.index(depth) // depth_bucket_step == d
        ]
        for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
    ]

    if normalization_type == "cardinality":
        bucketed_cardinalities_by_depth = [
            [
                cluster.cardinality
                for depth in clusters_by_depth
                for cluster in depth
                if clusters_by_depth.index(depth) // depth_bucket_step == d
            ]
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ]

    max_ratio = max([ratio for depth in bucketed_ratios_by_depth for ratio in depth])
    bins = numpy.arange(
        0, max_ratio + max_ratio / num_ratio_buckets, max_ratio / num_ratio_buckets
    )

    if normalization_type == "cluster":
        data = numpy.transpose(
            [
                (pandas.cut(ratios_set, bins=bins).value_counts() / len(ratios_set))[
                    ::-1
                ].to_list()
                for ratios_set in bucketed_ratios_by_depth
            ]
        )
    else:
        data = numpy.transpose(
            [
                insert_empty_bins(
                    bins,
                    pandas.Series(cards_set, index=pandas.cut(ratios_set, bins).to_list())
                    .groupby(level=0)
                    .sum()
                    / sum(cards_set),
                )[::-1]
                for ratios_set, cards_set in zip(
                    bucketed_ratios_by_depth, bucketed_cardinalities_by_depth
                )
            ]
        )

    return seaborn.heatmap(
        data=data,
        vmin=0,
        vmax=0.5,
        cmap="Reds",
        annot=False,
        linewidths=0.75,
        linecolor="white",
        ax=ax,
    )


# def _heat_radius(
#     clusters_by_depth: list[list[reports.ClusterReport]],
#     ax: pyplot.Axes,
#     ratio_index: int,
#     normalization_type: typing.Literal["cluster", "cardinality"] = "cardinality",
#     num_ratio_buckets: int = 50,
#     depth_bucket_step: int = 2,
# ):

#     bucketed_ratios_by_depth = [
#         [
#             cluster.ratios[ratio_index]
#             for depth in clusters_by_depth
#             for cluster in depth
#             if clusters_by_depth.index(depth) // depth_bucket_step == d
#         ]
#         for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
#     ]

#     if normalization_type == "cardinality":
#         bucketed_cardinalities_by_depth = [
#             [
#                 cluster.cardinality
#                 for depth in clusters_by_depth
#                 for cluster in depth
#                 if clusters_by_depth.index(depth) // depth_bucket_step == d
#             ]
#             for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
#         ]

#     max_ratio = max([ratio for depth in bucketed_ratios_by_depth for ratio in depth])
#     bins = numpy.arange(
#         0, max_ratio + max_ratio / num_ratio_buckets, max_ratio / num_ratio_buckets
#     )

#     if normalization_type == "cluster":
#         data = numpy.transpose(
#             [
#                 (pandas.cut(ratios_set, bins=bins).value_counts() / len(ratios_set))[
#                     ::-1
#                 ].to_list()
#                 for ratios_set in bucketed_ratios_by_depth
#             ]
#         )
#     else:
#         data = numpy.transpose(
#             [
#                 insert_empty_bins(
#                     bins,
#                     pandas.Series(cards_set, index=pandas.cut(ratios_set, bins).to_list())
#                     .groupby(level=0)
#                     .sum()
#                     / sum(cards_set),
#                 )[::-1]
#                 for ratios_set, cards_set in zip(
#                     bucketed_ratios_by_depth, bucketed_cardinalities_by_depth
#                 )
#             ]
#         )

#     return seaborn.heatmap(
#         data=data,
#         vmin=0,
#         vmax=0.5,
#         cmap="Reds",
#         annot=False,
#         linewidths=0.75,
#         linecolor="white",
#         ax=ax,
#     )



def _set_heat_lfd_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
):
    ax.set_xlabel("depth")
    ax.set_ylabel("local fractal dimension")

    ax.set_xticks(
        [d for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))],
        [
            f"{depth_bucket_step*d}"
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ],
        rotation=270,
    )

    max_lfd = max([cluster.lfd for depth in clusters_by_depth for cluster in depth])
    lfd_step = round(max_lfd / num_lfd_buckets, 1)

    ax.set_yticks(
        [d for d in range(num_lfd_buckets)],
        [f"{round(lfd_step*d, 2)}" for d in range(num_lfd_buckets, 0, -1)],
    )

    return


def _set_heat_parent_child_ratio_labels(
    clusters_by_depth: list[list[reports.ClusterReport]],
    ax: pyplot.Axes,
    ratio_name: typing.Literal[
        "cardinality", "radius", "lfd", "cardinality_ema", "radius_ema", "lfd_ema"
    ],
    ratio_index: int,
    num_lfd_buckets: int = 50,
    depth_bucket_step: int = 2,
):
    ax.set_xlabel("depth")
    ax.set_ylabel(f"{ratio_name}")

    ax.set_xticks(
        [d for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))],
        [
            f"{depth_bucket_step*d}"
            for d in range(math.ceil(len(clusters_by_depth) / depth_bucket_step))
        ],
        rotation=270,
    )

    max_ratio = max(
        [
            cluster.ratios[ratio_index]
            for depth in clusters_by_depth
            for cluster in depth
        ]
    )
    lfd_step = round(max_ratio / num_lfd_buckets, 1)

    ax.set_yticks(
        [d for d in range(num_lfd_buckets)],
        [f"{round(lfd_step*d, 2)}" for d in range(num_lfd_buckets, 0, -1)],
    )

    return


def plot_parent_child_ratios_vs_depth(
    mode: typing.Literal["violin", "heat"],
    tree: reports.TreeReport,
    clusters_by_depth: list[list[reports.ClusterReport]],
    show: bool,
    output_dir: pathlib.Path,
    ratio_name: typing.Literal[
        "cardinality", "radius", "lfd", "cardinality_ema", "radius_ema", "lfd_ema"
    ],
):
    figure: pyplot.Figure = pyplot.figure(figsize=(16, 10), dpi=300)
    title = ", ".join(
        [
            f"name = {tree.data_name}",
            f"metric = {tree.metric_name}",
            f"cardinality = {tree.cardinality}",
            f"dimensionality = {tree.dimensionality}",
            f"build_time = {tree.build_time:3.2e} (sec)",
        ]
    )
    figure.suptitle(title)

    ratios = {
        "cardinality": 0,
        "radius": 1,
        "lfd": 2,
        "cardinality_ema": 3,
        "radius_ema": 4,
        "lfd_ema": 5,
    }

    ax = (_violin_parent_child_ratio if mode == "violin" else _heat_ratio)(
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), ratios[ratio_name]
    )

    if mode == "violin":
        ax = _violin_parent_child_ratio(clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), ratios[ratio_name])
        _set_violin_labels(
            clusters_by_depth,
            ax,
            "depth",
            f"{ratio_name}" )
    else:
        ax = _heat_ratio(clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), ratios[ratio_name])
        _set_heat_ratio_labels(
            clusters_by_depth, ax, ratio_name, ratios[ratio_name], 50, 2
        )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"{mode}-{ratio_name}-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return


def plot_radius_vs_depth(
    mode: typing.Literal["violin", "heat"],
    tree: reports.TreeReport,
    clusters_by_depth: list[list[reports.ClusterReport]],
    show: bool,
    output_dir: pathlib.Path,
    radius_type: typing.Literal[
   "old_radius", "new_radius"],
):
    figure: pyplot.Figure = pyplot.figure(figsize=(16, 10), dpi=300)
    title = ", ".join(
        [
            f"name = {tree.data_name}",
            f"metric = {tree.metric_name}",
            f"cardinality = {tree.cardinality}",
            f"dimensionality = {tree.dimensionality}",
            f"build_time = {tree.build_time:3.2e} (sec)",
            f"{radius_type} radius",
        ]
    )
    figure.suptitle(title)


    # ax = (_violin_radius if mode == "violin" else _heat_ratio)(
    #     clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), radius_type
    # )

    if mode == "violin":
        ax = _violin(clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), radius_type)
        _set_violin_labels(
            clusters_by_depth,
            ax,
            "depth"
            f"{radius_type} radius",
            clamp_y = True, 
            clamping_function = "radii_comparison"
        )
    # else:
    #     _set_heat_ratio_labels(
    #         clusters_by_depth, ax, ratio_name, ratios[ratio_name], 50, 2
    #     )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"{mode}-{radius_type}-radius-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return

def plot_radii_ratio_vs_depth(
    mode: typing.Literal["violin", "heat", "line"],
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
            f"radii ratio",
        ]
    )
    figure.suptitle(title)

    (_violin if mode == "violin" else _line )(
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85)), "radii_ratio", "depth", "max"
    )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"{mode}-radii-ratio-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return



def plot_mean_radius_vs_depth(
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
            f"mean radius by depth",
        ]
    )
    figure.suptitle(title)


    ax = _line_radius_comparison (
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85))
    )

    _set_line_labels(clusters_by_depth, ax, "depth", "radius")

    ax.legend()

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"line-mean-radius-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return

def plot_ratio_vs_depth(
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
            f"mean radii ratio by depth",
        ]
    )
    figure.suptitle(title)


    ax = _line (
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85))
    )

    ax.set_xlabel("depth")
    ax.set_ylabel("radii ratio")

    ax.set_xticks(
        [d for d in range(len(clusters_by_depth))],
        [f"{d}" for d, clusters in enumerate(clusters_by_depth)],
        rotation=270,
    )
    ax.set_ylim(0.0, 1.2)

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"line-mean-ratio-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return


def plot_scatter_radii_ratio_vs_cardinality(
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
            f"radius by cardinality",
        ]
    )
    figure.suptitle(title)


    ax = _scatter_radii_ratio_by_cardinality(
        clusters_by_depth, pyplot.axes((0.05, 0.1, 0.9, 0.85))
    )

    # ax.set_xlabel("cardinality")
    # ax.set_ylabel("radius")

    # x_high = clusters_by_depth[0][0].cardinality
    # max_radius = max(clusters_by_depth[0][0].old_radius, clusters_by_depth[0][0].radius)
    # y_high = int(math.ceil(max_radius / 1000.0)) * 1000 

    # ax.set_xticks(
    #     [d for d in range(0, x_high+1, 2000)],
    #     [f"{d}" for d in  range(0, y_high+1, 2000)],
    # )

    # ax.set_yticks(
    #     [d for d in range(0, y_high+1, 500)],
    #     [f"{d}" for d in  range(0, y_high+1, 500)],
    # )

    if show:
        pyplot.show()
    else:
        figure.savefig(
            output_dir.joinpath(
                f"scatter-cardinality-ratio-{tree.data_name}__{tree.metric_name}.png"
            ),
            dpi=300,
        )
    pyplot.close(figure)

    return