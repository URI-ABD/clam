import enum
import pathlib

import matplotlib.pyplot as plt

from .reports import Report


class Markers(str, enum.Enum):
    GreedySieve = "GreedySieve"
    Linear = "Linear"
    RepeatedRnn = "RepeatedRnn"
    Sieve = "Sieve"
    SieveSepCenter = "SieveSepCenter"
    Faiss = "Faiss"
    HNSW = "HNSW"
    MRPT = "MRPT"
    BruteForceBLAS = "BruteForceBLAS"

    def marker(self) -> str:
        """Return the marker for the algorithm."""
        if self == Markers.GreedySieve:
            m = "x"
        elif self == Markers.Linear:
            m = "."
        elif self == Markers.RepeatedRnn:
            m = "o"
        elif self == Markers.Sieve:
            m = "s"
        elif self == Markers.SieveSepCenter:
            m = "d"
        elif self == Markers.Faiss:
            m = ">"
        elif self == Markers.HNSW:
            m = "<"
        elif self == Markers.MRPT:
            m = "^"
        elif self == Markers.BruteForceBLAS:
            m = "v"
        else:
            raise ValueError(f"Unknown algorithm {self}")
        return m


def create_plots(
    json_path: pathlib.Path, make_title: bool, output_dir: pathlib.Path
) -> bool:
    """Plot the throughput of the Cakes search."""
    report = Report.from_json(json_path)
    df = report.to_pandas()

    for k, k_df in df.groupby("k"):
        name = f"{report.dataset}_{k}"
        output_path = output_dir.joinpath(f"{name}_scaling.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        for algorithm, algorithm_df in k_df.groupby("algorithm"):
            marker = Markers(str(algorithm)).marker()
            ax.plot(
                algorithm_df["scale"],
                algorithm_df["throughput"],
                label=algorithm,
                marker=marker,
            )

        if make_title:
            title = (
                f"Scaling {report.dataset}-{report.metric} ({report.base_cardinality}x"
                f"{report.dimensionality}) with error rate {report.error_rate} "
                f"and k={k}"
            )
            ax.set_title(title)

        ax.set_xlabel("Multiplier")
        ax.set_ylabel("Throughput (queries/s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(df["scale"].min() * 0.9, df["scale"].max() * 1.1)
        ax.set_ylim(df["throughput"].min() * 0.9, df["throughput"].max() * 1.1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)

    return True
