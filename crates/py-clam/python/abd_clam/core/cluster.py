"""This module contains the `Cluster` class."""

import math
import typing

import numpy

from ..utils import constants
from ..utils import helpers
from . import cluster_criteria
from . import space

logger = helpers.make_logger(__name__)

Ratios = tuple[float, float, float, float, float, float]
RATIO_NAMES = ["cardinality", "radius", "lfd"]
RATIO_NAMES.extend([f"{name}_ema" for name in RATIO_NAMES])


# TODO: Figure out a theoretically sound way to have this as part of the Dataset class.
SUB_SAMPLE_LIMIT = 100


class Cluster:
    """A `Cluster` represents a collection of `"nearby"` instances in a metric space.

    A `Cluster` is a node in a binary tree we call the `Cluster Tree`.

    Ideally, a user will instantiate a `Cluster` using the `new_root` method and
    chain calls to the `build` and `partition` methods to create the full tree.
    We also provide an `iterative_partition` method to avoid the inefficiencies
    of recursion in Python.

    The name of a `Cluster` is uniquely determined by its location in the tree.
    Names are generated as they would be for a huffman tree. The root cluster
    has the name `"1"`. A left-child appends a `"0"` and a right-child appends
    a `"1"` to the name of the parent.
    """

    __slots__ = [
        "__metric_space",
        "__indices",
        "__name",
        "__parent",
        "__arg_samples",
        "__arg_center",
        "__arg_radius",
        "__radius",
        "__lfd",
        "__ratios",
        "__children",
        "__ancestry",
        "__subtree",
        "__candidate_neighbors",
    ]

    @staticmethod
    def new_root(metric_space: space.Space) -> "Cluster":
        """Creates a new root cluster for the given metric space."""
        return Cluster(
            metric_space,
            indices=list(range(metric_space.data.cardinality)),
            name="1",
            parent=None,
        )

    def __init__(
        self,
        metric_space: space.Space,
        *,
        indices: list[int],
        name: str,
        parent: typing.Optional["Cluster"],
    ) -> None:
        """Initializes a new Cluster."""
        if len(indices) == 0:
            msg = "Cannot instantiate a cluster with an empty list of `indices`."
            raise ValueError(
                msg,
            )

        self.__metric_space = metric_space
        self.__indices = indices
        self.__name = name
        self.__parent: typing.Optional["Cluster"] = parent

        self.__arg_samples: typing.Union[list[int], constants.Unset] = constants.UNSET
        self.__arg_center: typing.Union[int, constants.Unset] = constants.UNSET
        self.__arg_radius: typing.Union[int, constants.Unset] = constants.UNSET
        self.__radius: typing.Union[float, constants.Unset] = constants.UNSET
        self.__lfd: typing.Union[float, constants.Unset] = constants.UNSET
        self.__ratios: typing.Union[Ratios, constants.Unset] = constants.UNSET
        self.__children: typing.Optional[tuple["Cluster", "Cluster"]] = None
        self.__ancestry: typing.Union[
            list["Cluster"],
            constants.Unset,
        ] = constants.UNSET
        self.__subtree: typing.Union[
            list[list["Cluster"]],
            constants.Unset,
        ] = constants.UNSET
        self.__candidate_neighbors: typing.Union[
            dict["Cluster", float],
            constants.Unset,
        ] = constants.UNSET

    def __eq__(self, other: "Cluster") -> bool:  # type: ignore[override]
        """Clusters are equal if they have the same name."""
        return self.__name == other.__name

    def __lt__(self, other: "Cluster") -> bool:
        """Clusters are ordered by depth and then by name."""
        if self.depth == other.depth:
            return self.__name < other.__name
        return self.depth < other.depth

    def __str__(self) -> str:
        """Returns a string representation of the cluster."""
        return self.__name

    def __repr__(self) -> str:
        """Returns a string representation of the cluster."""
        return f"{self.metric_space} :: Cluster {self.__name}"

    def __hash__(self) -> int:
        """Hash the name of the cluster."""
        return hash(repr(self.__name))

    @property
    def metric_space(self) -> space.Space:
        """The metric space in which the cluster exists."""
        return self.__metric_space

    @property
    def indices(self) -> list[int]:
        """The indices of instances in the cluster."""
        return self.__indices

    @property
    def name(self) -> str:
        """The name (i.e. location) of the cluster in the tree."""
        return self.__name

    @property
    def depth(self) -> int:
        """The depth of the cluster in the tree."""
        return len(self.__name) - 1

    @property
    def cardinality(self) -> int:
        """The number of instances in the cluster."""
        return len(self.__indices)

    @property
    def arg_samples(self) -> list[int]:
        """The indices of sub-sampled instances used to find the center."""
        if self.__arg_samples is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__arg_samples  # type: ignore[return-value]

    @property
    def arg_center(self) -> int:
        """Index of the cluster center."""
        if self.__arg_center is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__arg_center  # type: ignore[return-value]

    @property
    def center(self) -> typing.Any:  # noqa: ANN401
        """The (approximate) geometric median of the cluster."""
        if self.__arg_center is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__metric_space.data[self.__arg_center]  # type: ignore[index]

    @property
    def arg_radius(self) -> int:
        """Index of the farthest instance from the center."""
        if self.__arg_radius is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__arg_radius  # type: ignore[return-value]

    @property
    def radius(self) -> float:
        """Distance form the center to the farthest instance in the cluster."""
        if self.__radius is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__radius  # type: ignore[return-value]

    @property
    def is_singleton(self) -> bool:
        """Returns `True` if the cluster has zero radius.

        i.e. if the cluster has only one instance or all instances are identical.
        """
        if self.__radius is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__radius == 0

    @property
    def lfd(self) -> float:
        """The local fractal dimension of the cluster."""
        if self.__lfd is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__lfd  # type: ignore[return-value]

    @property
    def parent(self) -> typing.Optional["Cluster"]:
        """The parent of this cluster in the tree."""
        return self.__parent

    @property
    def ratios(self) -> Ratios:
        """The six cluster ratios used for mata machine learning in CHAODA.

        To build a correct set of ratios, the user need to call the
        `normalize_ratios` method on the root of a tree.
        """
        if self.__ratios is constants.UNSET:
            msg = "Please call `build` on the cluster before using this property."
            raise ValueError(
                msg,
            )
        return self.__ratios  # type: ignore[return-value]

    @property
    def is_leaf(self) -> bool:
        """Returns `True` if this cluster is a leaf in the tree."""
        return self.__children is None

    @property
    def children(self) -> typing.Optional[tuple["Cluster", "Cluster"]]:
        """The two children of this cluster."""
        return self.__children

    @property
    def max_leaf_depth(self) -> int:
        """The depth of the deepest leaf in the `subtree` of this cluster."""
        return self.subtree[-1][0].depth

    @property
    def left_child(self) -> "Cluster":
        """The left child of this cluster."""
        if self.__children is None:
            msg = "This Cluster is a leaf and has no children."
            raise ValueError(msg)
        return self.children[0]  # type: ignore[index]

    @property
    def right_child(self) -> "Cluster":
        """The right child of this cluster."""
        if self.__children is None:
            msg = "This Cluster is a leaf and has no children."
            raise ValueError(msg)
        return self.children[1]  # type: ignore[index]

    @property
    def subtree(self) -> list[list["Cluster"]]:
        """Returns a list of lists of all clusters in the subtree of this cluster.

        Each inner list contains clusters at the same depth.
        The subtree includes the cluster itself.
        """
        if self.__subtree is constants.UNSET:
            if self.is_leaf:
                self.__subtree = [[self]]
            else:
                left_subtree = self.left_child.subtree
                right_subtree = self.right_child.subtree

                if len(left_subtree) < len(right_subtree):
                    left_subtree.extend(
                        [] for _ in range(len(right_subtree) - len(left_subtree))
                    )

                if len(right_subtree) < len(left_subtree):
                    right_subtree.extend(
                        [] for _ in range(len(left_subtree) - len(right_subtree))
                    )

                self.__subtree = [[self]] + [
                    (left_layer + right_layer)
                    for left_layer, right_layer in zip(left_subtree, right_subtree)
                ]

        self.__subtree = [
            layer
            for layer in self.__subtree  # type: ignore[union-attr]
            if len(layer) > 0
        ]

        return self.__subtree

    @property
    def num_descendants(self) -> int:
        """Returns the number of Clusters in the subtree of this Cluster."""
        return sum(len(_c) for _c in self.subtree)

    @property
    def ancestry(self) -> list["Cluster"]:
        """Returns the path through the tree from the root to this Cluster."""
        if self.__ancestry is constants.UNSET:
            if self.__parent is None:
                self.__ancestry = []
            else:
                self.__ancestry = [*self.__parent.ancestry, self.__parent]
        return self.__ancestry  # type: ignore[return-value]

    @property
    def candidate_neighbors(self) -> dict["Cluster", float]:
        """Find clusters in the tree that could be neighbors.

        Or whose descendants could be neighbors, of this cluster or of its
        descendants if any subset of them were selected to build a `Graph`.

        This property helps optimize the process of finding edges when building
        `Graph`s.
        """
        if self.__candidate_neighbors is constants.UNSET:
            if self.__parent is None:
                self.__candidate_neighbors = {self: 0.0}
            else:
                candidates = self.__parent.candidate_neighbors
                radius = max(self.radius, candidates[self.__parent])

                non_leaf_candidates = [c for c in candidates if not c.is_leaf]
                if len(non_leaf_candidates) > 0:
                    children = [
                        child
                        for c in non_leaf_candidates
                        for child in c.children  # type: ignore[union-attr]
                    ]
                    arg_centers = [c.arg_center for c in children]
                    distances = list(
                        map(
                            float,
                            self.__metric_space.distance_one_to_many(
                                self.arg_center,
                                arg_centers,
                            ),
                        ),
                    )
                    candidates.update(
                        {
                            c: d
                            for c, d in zip(children, distances)
                            if d <= radius + c.radius * (1 + math.sqrt(2))
                        },
                    )

                candidates[self] = radius
                self.__candidate_neighbors = candidates

        return self.__candidate_neighbors  # type: ignore[return-value]

    def is_ancestor_of(self, other: "Cluster") -> bool:
        """Returns `True` if this cluster is an ancestor of `other`."""
        return (self.depth < other.depth) and (
            self.__name == other.__name[: len(self.__name)]
        )

    def is_descendent_of(self, other: "Cluster") -> bool:
        """Returns `True` if this cluster is a descendent of `other`."""
        return other.is_ancestor_of(self)

    def build(self) -> "Cluster":
        """Calculates and sets important properties of the Cluster.

        These include:
            - arg_samples
            - arg_center
            - center
            - arg_radius
            - radius
            - lfd (local fractal dimension)
            - ratios (before normalization).

        Returns:
            The modified `Cluster` with after calculating essential properties.
        """
        logger.debug(f"Building cluster {self.__name} ...")

        if self.cardinality < SUB_SAMPLE_LIMIT:
            self.__arg_samples = self.__indices.copy()
        else:
            n = int(math.sqrt(self.cardinality))
            self.__arg_samples = self.__metric_space.choose_unique(n, self.__indices)

        sample_distances = numpy.sum(
            self.__metric_space.distance_pairwise(self.__arg_samples),
            axis=0,
        )
        self.__arg_center = self.__arg_samples[numpy.argmin(sample_distances)]

        center_distances = self.__metric_space.distance_one_to_many(
            self.__arg_center,  # type: ignore[arg-type]
            self.__indices,
        )
        farthest = numpy.argmax(center_distances)
        self.__arg_radius = self.__indices[farthest]
        self.__radius = float(center_distances[farthest])

        if self.is_singleton:
            self.__lfd = 1
        else:
            half_count = sum(d <= (self.__radius / 2) for d in center_distances)
            if half_count == 0:
                msg = "half_count for singleton cluster was zero."
                raise ValueError(msg)
            if half_count == 1:
                self.__lfd = self.cardinality
            else:
                self.__lfd = self.cardinality / (math.log2(half_count))

        # Calculate cluster ratios
        if self.parent is None:
            self.__ratios = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        else:
            c = self.cardinality / self.parent.cardinality
            r = self.__radius / self.parent.radius
            l = self.__lfd / self.parent.lfd  # noqa: E741

            _, _, _, pc_, pr_, pl_ = self.parent.ratios
            c_ = helpers.next_ema(c, pc_)
            r_ = helpers.next_ema(r, pr_)
            l_ = helpers.next_ema(l, pl_)

            self.__ratios = (c, r, l, c_, r_, l_)

        return self

    def distance_to_other(self, other: "Cluster") -> float:
        """Compute the distance between the centers of the two Clusters."""
        return self.distance_to_indexed_instance(
            other.__arg_center,  # type: ignore[arg-type]
        )

    def distance_to_indexed_instance(self, index: int) -> float:
        """Compute the distance from the center of the Cluster to the instance."""
        return self.__metric_space.distance_one_to_one(
            self.__arg_center,  # type: ignore[arg-type]
            index,
        )

    def distance_to_instance(self, instance: typing.Any) -> float:  # noqa: ANN401
        """Compute the distance from the center of the Cluster to the instance."""
        return self.__metric_space.distance_metric.one_to_one(self.center, instance)

    def __can_be_partitioned(
        self,
        criteria: typing.Sequence[cluster_criteria.ClusterCriterion],
    ) -> bool:
        """Returns `True` if the cluster can be partitioned."""
        if isinstance(self.__arg_samples, constants.Unset):
            msg = (
                "Please call `build` on this Cluster before calling "
                "`partition` or `iterative_partition`."
            )
            raise ValueError(
                msg,
            )

        # Cannot partition a cluster with zero radius.
        # Every criterion must evaluate to True.
        return (not self.is_singleton) and all(c(self) for c in criteria)

    def iterative_partition(
        self,
        criteria: typing.Sequence[cluster_criteria.ClusterCriterion],
    ) -> "Cluster":
        """Iteratively partitions the cluster to leaves.

        This iteratively builds the tree in a breadth-first manner, avoiding
        the inefficiencies of recursion in Python.
        """
        if not self.__can_be_partitioned(criteria):
            return self

        layers: list[list["Cluster"]] = [[self]]

        while True:
            partitionable = [c for c in layers[-1] if c.__can_be_partitioned(criteria)]
            if len(partitionable) == 0:
                break

            logger.info(
                f"Partitioning {len(partitionable)} clusters at depth "
                f"{partitionable[0].depth} ...",
            )

            [c.partition(criteria, recursive=False) for c in partitionable]
            layers.append(
                [
                    child
                    for c in partitionable
                    for child in c.children  # type: ignore[union-attr]
                ],
            )

        self.__subtree = layers

        return self

    def partition(
        self,
        criteria: typing.Sequence[cluster_criteria.ClusterCriterion],
        recursive: bool = True,
    ) -> "Cluster":
        """Tries to partition this cluster into two child clusters.

        If the cluster gets partitioned, then the `children` property will be
        assigned as a tuple of the left and right child clusters.

        `partition` starts by selecting two highly separated instances from the
        cluster. These are the left and right poles. The remaining instances
        are then split into two sets. Instances closer to the left pole are in
        the left split and instances closer to the right pole are in the right
        split. The left and right sets, along with their respective poles, are
        each used to instantiate the child clusters. Each child cluster is then
        built and, optionally, partitioned with the same `criteria`.

        Args:
            criteria: A list of rules for partitioning a cluster. Each criterion
             will be called with `self` as the only argument. If any criterion
             returns `False`, then this cluster will not be partitioned.
            recursive: Whether to recursively partition child clusters.

        Returns:
            The modified cluster.
        """
        if not self.__can_be_partitioned(criteria):
            return self

        # The farthest instance from the center is used as one of the poles.
        left_pole = self.arg_radius
        remaining_indices = [i for i in self.__indices if i != left_pole]
        left_distances = self.__metric_space.distance_one_to_many(
            left_pole,
            remaining_indices,
        )

        # The farthest instance from the left pole is used as the right pole.
        arg_right = int(numpy.argmax(left_distances))
        right_pole: int = remaining_indices[arg_right]

        if len(remaining_indices) > 1:
            remaining_indices = [i for i in remaining_indices if i != right_pole]

            # For every instance that is not a pole, compute its distance to each pole.
            left_distances = numpy.asarray(
                [d for i, d in enumerate(left_distances) if i != arg_right],
                dtype=left_distances.dtype,
            )
            right_distances = self.__metric_space.distance_one_to_many(
                right_pole,
                remaining_indices,
            )

            is_closer_to_left_pole = [
                ld <= rd for ld, rd in zip(left_distances, right_distances)
            ]

            # The remaining instances are split by proximity to either pole.
            left_indices = [left_pole] + [
                i for i, b in zip(remaining_indices, is_closer_to_left_pole) if b
            ]
            right_indices = [right_pole] + [
                i for i, b in zip(remaining_indices, is_closer_to_left_pole) if not b
            ]

        else:
            left_indices = [i for i in self.__indices if i != right_pole]
            right_indices = [right_pole]

        # The children will be biased so that the cardinality of the left child
        # is no smaller than that of the right child.
        left_indices, right_indices = (
            (left_indices, right_indices)
            if len(left_indices) > len(right_indices)
            else (right_indices, left_indices)
        )

        # Recursively build and partition the left and right children.
        left_child = Cluster(
            self.__metric_space,
            indices=left_indices,
            name=self.__name + "0",
            parent=self,
        ).build()

        right_child = Cluster(
            self.__metric_space,
            indices=right_indices,
            name=self.__name + "1",
            parent=self,
        ).build()

        if recursive:
            left_child = left_child.partition(criteria)
            right_child = right_child.partition(criteria)

        self.__children = (left_child, right_child)

        return self

    def normalize_ratios(
        self,
        mode: helpers.NormalizationMode = "gaussian",
    ) -> "Cluster":
        """Normalize the cluster ratios in the `subtree` of this cluster.

        This method is intended to only be called once and only on the root.
        The user is responsible for this contract.

        Args:
            mode: Normalization method to use. Must be one of:
                - 'linear',
                - 'gaussian', or
                - 'sigmoid'.

        Returns:
            The modified cluster after setting the normalized ratios.
        """
        clusters = [c for layer in self.subtree for c in layer]
        ratios_array = numpy.asarray([c.ratios for c in clusters], dtype=numpy.float32)
        if ratios_array.shape != (len(clusters), len(RATIO_NAMES)):
            msg = (
                f"Expected ratios array to have shape "
                f"{len(clusters), len(RATIO_NAMES)}. "
                f"Got {ratios_array.shape} instead."
            )
            raise ValueError(msg)

        ratios_array = helpers.normalize(ratios_array, mode)

        for i, c in enumerate(clusters):
            c.__ratios = tuple(  # type: ignore[assignment]
                map(float, ratios_array[i, :]),
            )

        return self

    def add_instance(self, index: int) -> list["Cluster"]:
        """Add an instance to the tree after having added it to the metric space.

        This method may only be called on a root cluster. The primary use
        of this method is when the full dataset does not fit in memory. In such
        a case, the user can create a tree on a subset of the full data.
        Instances in the complement of the subset can then be added to the tree.

        The instance is recursively added to the child whose center is closer.

        This method will almost certainly invalidate cluster properties such as
        `center` and `radius`. It is upon the user to rebuild clusters after
        adding instances.

        Args:
            index: of the instance in the metric space.

        Returns:
            The list of clusters, from root to leaf, to which the instance was
             added.
        """
        # TODO: Figure out how and where an instance will be added to the metric space.

        if self.depth != 0:
            msg = "Cannot add an instance to a non-root Cluster."
            raise ValueError(msg)

        return sorted(self.__add_instance(index))

    def __add_instance(self, index: int) -> list["Cluster"]:
        """Recursively add the indexed instance to the tree.

        Appends `index` to `__indices` after addition.
        """
        result = [self]

        if not self.is_leaf:
            distance_to_left = self.left_child.distance_to_indexed_instance(index)
            distance_to_right = self.right_child.distance_to_indexed_instance(index)

            if distance_to_left <= distance_to_right:
                result.extend(self.left_child.__add_instance(index))
            else:
                result.extend(self.right_child.__add_instance(index))

        self.__indices.append(index)

        return result


__all__ = [
    "Cluster",
]
