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
    metric: str
    base_cardinality: int
    dimensionality: int
    num_queries: int
    error_rate: float
    ks: list[int]
    csv_path: pathlib.Path = pathlib.Path(".").resolve()

    @staticmethod
    def from_json(json_path: pathlib.Path) -> "Report":
        """Load the report from a JSON file."""
        with json_path.open("r") as json_file:
            contents: dict[str, typing.Any] = json.load(json_file)
            contents["csv_path"] = json_path.parent.joinpath(contents.pop("csv_name"))
            return Report(**contents)

    def to_pandas(self) -> pandas.DataFrame:
        """Read the CSV file into a pandas DataFrame."""
        return pandas.read_csv(self.csv_path)


def plot_throughput(json_path: pathlib.Path, output_dir: pathlib.Path) -> bool:
    """Plot the throughput of the Cakes search."""
    report = Report.from_json(json_path)
    df = report.to_pandas()

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

    title_pieces = [
        f"Dataset: {report.dataset}",
        f"Metric: {report.metric}",
        f"Error Rate: {report.error_rate}",
        f"Base Cardinality: {report.base_cardinality}",
        f"Dimensionality: {report.dimensionality}",
    ]
    ax.set_title("Scaling Benchmarks: " + ", ".join(title_pieces))
    ax.set_xlabel("Multiplier")
    ax.set_ylabel("Throughput (queries/s)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(df["scale"].unique())
    ax.set_xticklabels(df["scale"].unique())
    ax.set_xlim(df["scale"].min() * 0.9, df["scale"].max() * 1.1)
    ax.set_ylim(df["throughput"].min() * 0.9, df["throughput"].max() * 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    return True
