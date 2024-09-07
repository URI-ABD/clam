"""Exploring the logs of long Clustering runs."""

import logging
import pathlib
import re

logger = logging.getLogger(__name__)


def count_clusters(file_path: pathlib.Path) -> list[tuple[bool, int, int]]:
    """Count the number of lines in a file that contain information about clusters."""
    pattern = re.compile(
        r"^(?P<status>Starting|Finished) `par_partition` of a cluster at depth (?P<depth>\d+), with (?P<cardinality>\d+) instances\.$",  # noqa: E501
    )

    cluster_counts = []

    with file_path.open("r") as file:
        for line in file:
            if match := pattern.match(line.strip()):
                status = match.group("status") == "Finished"
                depth = int(match.group("depth"))
                cardinality = int(match.group("cardinality"))
                cluster_counts.append((status, depth, cardinality))

    msg = f"Found {len(cluster_counts)} clusters in {file_path}."
    logger.info(msg)

    return cluster_counts


def clusters_by_depth(
    clusters: list[tuple[bool, int, int]],
) -> list[tuple[int, tuple[tuple[int, int], tuple[int, int]]]]:
    """Count the number of clusters by depth."""
    depth_counts: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    for status, depth, cardinality in clusters:
        (s_freq, s_count), (f_freq, f_count) = depth_counts.get(depth, ((0, 0), (0, 0)))
        if status:
            f_freq += 1
            f_count += cardinality
        else:
            s_freq += 1
            s_count += cardinality
        depth_counts[depth] = (s_freq, s_count), (f_freq, f_count)

    return sorted(depth_counts.items())


def wrangle_logs(log_path: pathlib.Path) -> None:
    """Wrangle the logs of long Clustering runs."""
    msg = f"Analyzing {log_path}..."
    logger.info(msg)

    clusters = count_clusters(log_path)
    progress = clusters_by_depth(clusters)

    gg_car = 989_002
    for depth, ((s_freq, s_card), (f_freq, f_card)) in progress:
        if depth % 256 < 50:
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
