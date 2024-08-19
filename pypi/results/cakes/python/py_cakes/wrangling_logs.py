"""Exploring the logs of long Clustering runs."""

import logging
import pathlib
import re

logger = logging.getLogger(__name__)


def count_clusters(file_path: pathlib.Path) -> dict[str, int]:
    """Count the number of lines in a file that contain information about clusters."""
    pattern = re.compile(
        r"^(?P<status>Starting|Finished) `par_partition` of a cluster at depth (?P<depth>\d+), with (?P<instances>\d+) instances\.$",  # noqa: E501
    )

    cluster_counts: dict[str, int] = {}

    with file_path.open("r") as file:
        for line in file:
            if match := pattern.match(line.strip()):
                key = f"{match.group('status')} {match.group('depth')} {match.group('instances')}"
                cluster_counts[key] = cluster_counts.get(key, 0) + 1

    total_clusters = sum(cluster_counts.values())
    msg = f"Found {total_clusters} clusters in {file_path}."
    logger.info(msg)

    return cluster_counts


def clusters_by_depth(clusters: dict[str, int]) -> dict[str, tuple[int, int]]:
    """Count the number of clusters by depth."""
    depth_counts: dict[str, tuple[int, int]] = {}

    for k, v in clusters.items():
        [status, depth, instances] = k.split()

        key = f"{status} {depth}"
        (freq, count) = depth_counts.get(key, (0, 0))
        depth_counts[key] = (freq + v, count + int(instances))

    return depth_counts


def track_progress(
    depth_counts: dict[str, tuple[int, int]],
) -> dict[int, tuple[tuple[int, int], tuple[int, int]]]:
    """Track the progress of the clustering process by depth."""
    # The key is the depth and the value is a tuple with the number of clusters
    # that finished with their total instances and started with their total
    # instances at that depth.
    progress: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    for k, (freq, count) in depth_counts.items():
        [status, depth_str] = k.split()
        depth = int(depth_str)

        ((finished, f_count), (started, s_count)) = progress.get(depth, ((0, 0), (0, 0)))

        if status == "Finished":
            finished += freq
            f_count += count
        else:
            started += freq
            s_count += count

        progress[depth] = ((finished, f_count), (started, s_count))

    return progress
