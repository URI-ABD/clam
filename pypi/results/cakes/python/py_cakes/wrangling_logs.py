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
