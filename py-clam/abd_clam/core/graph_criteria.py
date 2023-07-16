"""Graph criteria for selecting clusters from a tree to make graphs."""

import abc
import heapq
import typing

import numpy

from ..utils import helpers
from . import cluster

logger = helpers.make_logger(__name__)


class GraphCriterion(abc.ABC):
    """Criterion for selecting clusters from a tree to make graphs.

    A `GraphCriterion` can be called on the `root` cluster of a tree to
    select a set of clusters from the tree. These clusters can be used to
    create a `Graph`. Subclasses much implement the `select` method. The
    abstract `GraphCriterion` will verify that the set of selected clusters
    obey the graph invariant. See the documentation for `Graph` for details on
    the graph invariant.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the criterion."""
        pass

    @abc.abstractmethod
    def select(self, root: cluster.Cluster) -> set[cluster.Cluster]:
        """Selects a set of clusters from the tree rooted at `root`."""
        pass

    @staticmethod
    def assert_invariant(root: cluster.Cluster, selected: set[cluster.Cluster]) -> None:
        """Verifies that the set of selected clusters obey the graph invariant."""
        for c in selected:
            if any(c.is_ancestor_of(other) for other in selected):
                msg = "A cluster and its ancestor were both selected."
                raise ValueError(msg)

        indices = {i for c in selected for i in c.indices}

        if len(indices) != root.cardinality:
            msg = (
                f"There was a mis-match in the number of instances that were selected. "
                f"The selected clusters have {len(indices)} instance but the "
                f"root has {root.cardinality}."
            )
            raise ValueError(msg)

    def __call__(self, root: cluster.Cluster) -> set[cluster.Cluster]:
        """Selects a set of clusters from the tree rooted at `root`."""
        selected = self.select(root)
        self.assert_invariant(root, selected)
        return selected


class Layer(GraphCriterion):
    """Selects the layer at the specified depth, -1 means go to leaves."""

    def __init__(self, depth: int) -> None:
        """Initializes a `Layer` criterion."""
        if depth < -1:
            msg = f"expected a '-1' or a non-negative depth. got: {depth}"
            raise ValueError(msg)
        self.depth = depth

    @property
    def name(self) -> str:
        """Returns the name of the criterion."""
        return f"Layer_{self.depth}"

    def select(self, root: cluster.Cluster) -> set[cluster.Cluster]:
        """Selects the layer at the specified depth, -1 means go to leaves."""
        if self.depth == -1:
            return {c for layer in root.subtree for c in layer if c.is_leaf}

        selected = {
            c for layer in root.subtree[: self.depth] for c in layer if c.is_leaf
        }
        selected.update(root.subtree[self.depth])

        return selected


class PropertyThreshold(GraphCriterion):
    """Selects clusters when the given property crosses the given percentile."""

    def __init__(
        self,
        value: typing.Literal["cardinality", "radius", "lfd"],
        percentile: float,
        mode: typing.Literal["above", "below"],
    ) -> None:
        """Initializes a `PropertyThreshold` criterion.

        During a BFT of the tree, this keeps track of the given cluster
        property. A cluster is selected if any of the following is true:
            - it is a leaf, or
            - it qualifies by `mode`, or
            - both children qualify by `mode`.

        If `mode` is 'above', a cluster qualifies if the property goes above the
        `percentile`. If `mode` is 'below', a cluster qualifies if the property
        goes below the `percentile`.

        Args:
            value: The cluster property to keep track of. Must be one of
             'cardinality', 'radius', or 'lfd'.
            percentile: the percentile, in the (0, 100) range, to be crossed.
            mode: select a cluster when `value` is 'above' or 'below' the
             `percentile`.
        """
        if 0.0 < percentile < 100.0:
            self.percentile: float = percentile
        else:
            msg = (
                f"percentile must be in the (0, 100) range. "
                f"Got {percentile:.2f} instead."
            )
            raise ValueError(msg)

        self.value = value

        if mode == "above":
            self.qualifies = lambda c, v: getattr(c, self.value) > v
        elif mode == "below":
            self.qualifies = lambda c, v: getattr(c, self.value) < v
        else:
            msg = f"mode must be 'above' or 'below'. Got {mode} instead."
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Returns the name of the criterion."""
        return f"PropertyThreshold_{self.value}_{self.name}_{self.percentile:.2f}"

    def select(self, root: cluster.Cluster) -> set[cluster.Cluster]:
        """Selects clusters when the given property crosses the given threshold."""
        threshold = float(
            numpy.percentile(
                [
                    getattr(c, self.value)
                    for layer in root.subtree
                    for c in layer
                    if c.cardinality > 1
                ],
                self.percentile,
            ),
        )
        selected: set[cluster.Cluster] = set()
        frontier: set[cluster.Cluster] = {root}
        while frontier:
            c = frontier.pop()
            if (
                c.is_leaf  # select the leaves
                or self.qualifies(c, threshold)  # if cluster qualifies
                or (
                    self.qualifies(c.left_child, threshold)
                    and self.qualifies(c.right_child, threshold)
                )
            ):
                selected.add(c)
            else:  # add children to frontier
                frontier.update(c.children)  # type: ignore[arg-type]
        return selected


class MetaMLSelect(GraphCriterion):
    """Uses the scoring function from a trained meta-ml model to select clusters."""

    def __init__(
        self,
        scorer: typing.Callable[[numpy.ndarray], float],
        name: typing.Optional[str] = None,
        min_depth: int = 4,
    ) -> None:
        """Initializes a `MetaMLSelect` criterion.

        Args:
            scorer: A function that takes the ratios of a cluster and returns
            its predicted score. Higher scoring clusters are considered
            better than lower scoring clusters.
            name: The name of the criterion.
            min_depth: The minimum depth in the tree for a cluster to be
            selected.
        """
        if min_depth < 1:
            msg = "min-depth must be a positive integer."
            raise ValueError(msg)

        self.__name = scorer.__name__ if name is None else name
        self.scorer = lambda c: -scorer(numpy.asarray(c.ratios, dtype=numpy.float32))
        self.min_depth = min_depth

    @property
    def name(self) -> str:
        """Returns the name of the criterion."""
        return self.__name

    def select(self, root: cluster.Cluster) -> set[cluster.Cluster]:
        """Selects clusters using the scoring function."""
        tree = [c for layer in root.subtree for c in layer]

        candidate_clusters = [c for c in tree if c.depth >= self.min_depth]
        normalized_scores = list(
            map(
                float,
                helpers.normalize(
                    numpy.asarray(list(map(self.scorer, candidate_clusters))),
                    mode="gaussian",
                ),
            ),
        )

        heap = list(zip(normalized_scores, candidate_clusters))
        heapq.heapify(heap)

        selected = {c for c in tree if c.depth < self.min_depth and c.is_leaf}
        selected_indices = {i for c in selected for i in c.indices}

        while len(heap) > 0:
            _, c = heapq.heappop(heap)

            if len(selected_indices.intersection(set(c.indices))) > 0:
                continue
            selected.add(c)
            selected_indices.update(set(c.indices))

        return selected


__all__ = [
    "GraphCriterion",
    "Layer",
    "PropertyThreshold",
    "MetaMLSelect",
]
