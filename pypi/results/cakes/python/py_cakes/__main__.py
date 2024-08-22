"""CLI for the package."""

import logging
import pathlib

import typer

from py_cakes.wrangling_logs import clusters_by_depth
from py_cakes.wrangling_logs import count_clusters

logger = logging.getLogger("py_cakes")
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


app = typer.Typer()


@app.command()
def main(
    log_path: pathlib.Path = typer.Option(  # noqa: B008
        ...,
        help="Path to the log file to analyze.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """Main entry point."""
    msg = f"Analyzing {log_path}..."
    logger.info(msg)

    clusters = count_clusters(log_path)
    progress = clusters_by_depth(clusters)

    gg_car = 1_075_170
    for depth, ((s_freq, s_car), (f_freq, f_car)) in progress:
        msg = (
            f"Depth {depth:4d}: Started {s_freq:7d} clusters with {s_car:7d} instances, "
            f"finished {f_freq:7d} ({100 * f_freq / s_freq:3.2f}%) clusters with "
            f"{f_car:7d} ({100 * f_car / gg_car:3.2f}%) instances."
        )
        logger.info(msg)


if __name__ == "__main__":
    app()
