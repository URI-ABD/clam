"""Defines criteria for when to partition clusters."""

import abc

from ..utils import helpers
from . import cluster

logger = helpers.make_logger(__name__)


class ClusterCriterion(abc.ABC):
    """A rule to decide when a cluster can be partitioned.

    Subclasses must implement the `__call__` method to take a single `Cluster`
    and return a `bool` to indicate whether that cluster can be partitioned.

    If multiple criteria are used during `partition`, they must all return
    `True` to allow the cluster to be partitioned.
    """

    @abc.abstractmethod
    def __call__(self, c: "cluster.Cluster") -> bool:
        """Returns `True` if the cluster can be partitioned."""
        pass


class MaxDepth(ClusterCriterion):
    """Clusters with `depth` less than `max_depth` may be partitioned."""

    def __init__(self, max_depth: int) -> None:
        """Initializes the criterion with the given `max_depth`."""
        self.max_depth = max_depth

    def __call__(self, c: "cluster.Cluster") -> bool:
        """Returns `True` if the cluster is shallower than `max_depth`."""
        return c.depth < self.max_depth


class MinPoints(ClusterCriterion):
    """Clusters with `cardinality` greater than `min_points` may be partitioned."""

    def __init__(self, min_points: int) -> None:
        """Initializes the criterion with the given `min_points`."""
        self.min_points = min_points

    def __call__(self, c: "cluster.Cluster") -> bool:
        """Returns `True` if the cluster has more points than `min_points`."""
        return c.cardinality > self.min_points


class NotSingleton(ClusterCriterion):
    """Clusters can be partitioned so long as they are not singletons."""

    def __call__(self, c: "cluster.Cluster") -> bool:
        """Returns `True` if the cluster is not a singleton."""
        return not c.is_singleton


__all__ = [
    "ClusterCriterion",
    "MaxDepth",
    "MinPoints",
]
