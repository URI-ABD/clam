"""The plotting package."""

import enum
import logging
import math
import pathlib
import pprint
import typing

import pandas
import seaborn
import tqdm
from matplotlib import pyplot

from . import criterion
from . import report

logger = logging.getLogger(__file__)
logger.setLevel("INFO")


ANN_DATASETS = {
    "deep-image": ("cosine", 9_990_000, 96),
    "fashion-mnist": ("euclidean", 60_000, 784),
    "gist": ("euclidean", 1_000_000, 960),
    "glove-25": ("cosine", 1_183_514, 25),
    "glove-50": ("cosine", 1_183_514, 50),
    "glove-100": ("cosine", 1_183_514, 100),
    "glove-200": ("cosine", 1_183_514, 200),
    "kosarak": ("jaccard", 75_962, 27_983),
    "mnist": ("euclidean", 60_000, 784),
    "nytimes": ("cosine", 290_000, 256),
    "sift": ("euclidean", 1_000_000, 128),
    "lastfm": ("cosine", 292_385, 65),
}


class Mode(str, enum.Enum):
    """The mode to run the application in."""

    Reports = "reports"
    Criterion = "criterion"

    def plot(self, inp_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
        """Make plots from reports."""
        # if self == Mode.Reports:
        raise NotImplementedError

    def throughput(self, inp_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
        """Print the throughput instead of plotting the elapsed time."""
        if self == Mode.Reports:
            lines = [
                "data_name,num_shards,algorithm,k,throughput,factor_over_linear",
            ]
            reports = load_reports(inp_dir)
            for data_name, d_reports in reports.items():
                for num_shards, r_reports in d_reports.items():
                    for r in r_reports:
                        line = list(
                            map(
                                str,
                                [
                                    data_name,
                                    num_shards,
                                    r.algorithm,
                                    r.k,
                                    f"{r.throughput:.3e}",
                                    f"{r.throughput / r.linear_throughput:.3e}",
                                ],
                            ),
                        )
                        lines.append(",".join(line))

            out_dir.joinpath("throughput.csv").write_text("\n".join(lines))

        elif self == Mode.Criterion:
            raise NotImplementedError
        else:
            msg = f"Unknown mode {self}."
            raise ValueError(msg)


def load_reports(inp_dir: pathlib.Path) -> dict[str, dict[int, list[report.Report]]]:
    """Load reports from a directory."""
    logger.info(f"Loading reports from {inp_dir} ...")

    report_paths = sorted(filter(lambda p: p.suffix == ".json", inp_dir.iterdir()))

    logger.info(f"Found {len(report_paths)} reports.")

    reports: dict[str, dict[int, list[report.Report]]] = {}
    for report_path in report_paths:
        r = report.Report.from_path(report_path)

        if r.data_name not in reports:
            reports[r.data_name] = {}

        if r.num_shards not in reports[r.data_name]:
            reports[r.data_name][r.num_shards] = []

        reports[r.data_name][r.num_shards].append(r)

    return reports


def plot_reports(inp_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    """Make plots from reports."""
    reports = load_reports(inp_dir)
    logger.info(f"Loaded reports for {len(reports)} data sets.")
    logger.info(f"Saving plots to {plots_dir} ...")

    for data_name, d_reports in reports.items():
        for num_shards, r_reports in d_reports.items():
            logger.info(f"Plotting {data_name} in {num_shards} shard(s) ...")

            first = r_reports[0]
            metric_name = first.metric_name
            cardinality = first.cardinality
            dimensionality = first.dimensionality
            num_queries = first.num_queries

            report_dfs = [
                pandas.DataFrame(
                    {
                        "k": [r.k] * num_queries,
                        "algorithm": [r.algorithm] * num_queries,
                        "time": list(map(math.log10, r.elapsed)),
                    },
                )
                for r in r_reports
            ]
            plot_violin(
                df=pandas.concat(report_dfs),
                data_name=data_name,
                metric_name=metric_name,
                num_shards=num_shards,
                cardinality=cardinality,
                dimensionality=dimensionality,
                plots_dir=plots_dir,
            )


def plot_criterion(inp_dir: pathlib.Path, plots_dir: pathlib.Path) -> None:
    """Make plots from criterion benchmarks."""
    dataset_dirs = sorted(
        filter(lambda p: p.is_dir() and p.name != "report", inp_dir.iterdir()),
    )
    logger.info(f"Found benchmarks for {len(dataset_dirs)} datasets.")

    for dataset_dir in dataset_dirs:
        name = dataset_dir.name.split("-")[1:-1]
        data_name = "-".join(name)
        metric_name, cardinality, dimensionality = ANN_DATASETS[data_name]
        logger.info(f"Plotting {data_name} ...")

        group = criterion.ShardGroup.from_path(dataset_dir)
        for num_shards, algorithms in group.benches.items():
            logger.info(f"Plotting {data_name} in {num_shards} shard(s) ...")

            shard_dfs = []
            for algorithm in algorithms:
                for k in algorithm.ks:
                    shard_dfs.append(
                        pandas.DataFrame(
                            {
                                "k": [k.k] * k.num_queries,
                                "algorithm": [algorithm.algorithm] * k.num_queries,
                                "time": list(map(math.log10, k.elapsed)),
                            },
                        ),
                    )
            plot_violin(
                df=pandas.concat(shard_dfs),
                data_name=data_name,
                metric_name=metric_name,
                num_shards=num_shards,
                cardinality=cardinality,
                dimensionality=dimensionality,
                plots_dir=plots_dir,
            )


def plot_violin(  # noqa: PLR0913
    df: pandas.DataFrame,
    data_name: str,
    metric_name: str,
    num_shards: int,
    cardinality: int,
    dimensionality: int,
    plots_dir: typing.Optional[pathlib.Path] = None,
) -> None:
    """Plot a violin plot of the given data frame."""
    fig: pyplot.Figure = pyplot.figure(figsize=(14, 7), dpi=128)

    ax: pyplot.Axes = seaborn.violinplot(
        data=df,
        x="algorithm",
        y="time",
        hue="k",
        linewidth=0.1,
        cut=0,
    )

    ax.set_ylabel("Time per Query (10 ^ seconds)")
    ax.set_title(
        f"{data_name} - {metric_name} - {num_shards} shard(s) - "
        f"({cardinality} x {dimensionality}) shape",
    )

    fig.tight_layout()

    if plots_dir is None:
        pyplot.show()
    else:
        plot_name = f"{data_name}_{num_shards}.png"
        fig.savefig(plots_dir / plot_name, dpi=128)
