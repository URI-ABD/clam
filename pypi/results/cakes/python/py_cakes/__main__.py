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

    gg_car = 1_074_170
    for depth, ((s_freq, s_card), (f_freq, f_card)) in progress:
        if depth % 256 < 16:
            lines = [
                "",
                f"Depth {depth:4d}:",
                f"Clusters:  Started {s_freq:7d}, finished {f_freq:7d}. {100 * f_freq / s_freq:3.2f}%).",  # noqa: E501
                f"Instances: Started {s_card:7d}, finished {f_card:7d}. {100 * f_card / s_card:3.2f}% of started, {100 * f_card / gg_car:3.2f}% of total.",  # noqa: E501
            ]
            msg = "\n".join(lines)
            logger.info(msg)

    msg = f"Built (or building) tree with {len(progress)} depth."
    logger.info(msg)


if __name__ == "__main__":
    app()
