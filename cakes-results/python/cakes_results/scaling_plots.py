"""Plots for the scaling results of the Cakes searchgit status."""

import json
import pathlib
import typing

import matplotlib.pyplot as plt
import pandas
import pydantic


class Report(pydantic.BaseModel):
    """Report of the scaling results of the Cakes search."""

    dataset: str
    base_cardinality: int
    dimensionality: int
    num_queries: int
    scales: list[int]
    error_rate: float
    ks: list[int]

    # throughput: list[tuple[int, float, list[str, list[tuple[int, float]]]]]

    @staticmethod
    def from_json(json_path: pathlib.Path) -> tuple["Report", pandas.DataFrame]:
        """Load the report from a JSON file."""
        with json_path.open("r") as json_file:
            contents: dict[str, typing.Any] = json.load(json_file)
            throughput = contents.pop("throughput")
            report = Report(**contents)

        df: pandas.DataFrame = pandas.DataFrame(
            columns=["scale", "build_time", "algorithm", "k", "throughput"]
        )
        for scale, build_time, algorithms in throughput:
            for algorithm, k_throughput in algorithms:
                for k, throughput in k_throughput:
                    df.loc[len(df)] = {
                        "scale": scale + 1,
                        "build_time": build_time,
                        "algorithm": algorithm,
                        "k": k,
                        "throughput": throughput,
                    }

        return report, df


def plot_throughput(json_path: pathlib.Path, output_dir: pathlib.Path) -> None:
    """Plot the throughput of the Cakes search."""

    report, df = Report.from_json(json_path)

    name = f"{report.dataset}_{report.error_rate}"
    output_path = output_dir.joinpath(f"{name}_scaling.png")

    fig, ax = plt.subplots(figsize=(16, 10))
    for algorithm, algorithm_df in df.groupby("algorithm"):
        for k, k_df in algorithm_df.groupby("k"):
            ax.plot(
                k_df["scale"],
                k_df["throughput"],
                label=f"{algorithm} (k={k})",
                marker="x",
            )

    ax.set_title(f"Scaling of the Cakes search ({report.dataset}) with error rate {report.error_rate}")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Throughput (queries/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(df["scale"].unique())
    ax.set_xticklabels(df["scale"].unique())
    ax.set_xlim(0.9, df["scale"].max() * 1.1)
    ax.set_ylim(1, df["throughput"].max() * 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
