//! How a `Cluster` is partitioned into child clusters.

use crate::{Cluster, DistanceValue};

mod bipolar_split;
mod branching_factor;
mod max_split;
mod span_reduction_factor;

pub use branching_factor::BranchingFactor;
pub use max_split::MaxSplit;
pub use span_reduction_factor::SpanReductionFactor;

use bipolar_split::{BipolarSplit, InitialPole};

/// A type-alias for the span splits after applying a partition strategy to a slice of items.
pub type Splits<'a, Id, I> = Vec<(usize, &'a mut [(Id, I)])>;

/// How a `Cluster` is partitioned into child clusters.
///
/// The default `PartitionStrategy` will partition any cluster with more than one non-center item to make a binary tree whose leaves are all clusters with too
/// few items to be partitioned any further. This default behavior may later be changed after we benchmark and analyze various strategies.
///
/// The strategy is controlled by the `predicate`, `max_split`, `branching_factor`, and `span_reduction` fields. It proceeds as follows:
///   - Given some items, we first create a Cluster as a leaf.
///   - We then evaluate the `predicate` to determine whether the cluster should be partitioned. If the `predicate` is satisfied, we proceed to partition the
///     cluster; otherwise, we leave the cluster as a leaf.
///   - Next, if the `max_split` is not `None`, we use the [`MaxSplit`] enum to determine how many children to create. Otherwise, we proceed to
///     `branching_factor`.
///   - Next, if the `branching_factor` is not `Unbounded`, we use the the [`BranchingFactor`] enum to determine how many children to create. Otherwise, we
///     proceed to `span_reduction_factor`.
///   - Finally, we use the [`SpanReductionFactor`] enum to determine how many children to create.
///
/// Note that our implementation of `PartitionStrategy` may change in the future to improve performance or to add new features.
///
/// # Type Parameters
///
/// - `P`: A function: `&Cluster<T, A> -> bool` that determines whether a cluster should be partitioned into child clusters.
#[must_use]
#[derive(Debug, Clone, Copy)]
pub struct PartitionStrategy<P> {
    /// The predicate that determines whether a cluster should be partitioned into child clusters.
    pub(crate) predicate: P,
    /// See [`MaxSplit`].
    pub(crate) max_split: MaxSplit,
    /// See [`BranchingFactor`].
    pub(crate) branching_factor: BranchingFactor,
    /// See [`SpanReductionFactor`].
    pub(crate) span_reduction: SpanReductionFactor,
}

impl<P> std::fmt::Display for PartitionStrategy<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !matches!(self.max_split, MaxSplit::None) {
            write!(f, "MS({})", self.max_split)
        } else if !matches!(self.branching_factor, BranchingFactor::Unbounded) {
            write!(f, "BF({})", self.branching_factor)
        } else {
            write!(f, "SRF({})", self.span_reduction)
        }
    }
}

impl<T, A> Default for PartitionStrategy<fn(&Cluster<T, A>) -> bool> {
    fn default() -> Self {
        Self {
            predicate: |b: &Cluster<T, A>| b.cardinality > 2,
            max_split: MaxSplit::default(),
            branching_factor: BranchingFactor::Fixed(2),
            span_reduction: SpanReductionFactor::default(),
        }
    }
}

impl<T, A> PartitionStrategy<fn(&Cluster<T, A>) -> bool> {
    /// Creates a new `PartitionStrategy` that never partitions any cluster.
    pub fn never() -> Self {
        Self {
            predicate: |_| false,
            max_split: MaxSplit::default(),
            branching_factor: BranchingFactor::default(),
            span_reduction: SpanReductionFactor::default(),
        }
    }

    /// Changes the `PartitionStrategy` to always partition any cluster with a radius greater than the given threshold.
    pub fn with_radius_greater_than(self, threshold: T) -> PartitionStrategy<impl Fn(&Cluster<T, A>) -> bool>
    where
        T: PartialOrd,
    {
        self.with_predicate(move |c: &Cluster<T, A>| c.radius > threshold)
    }

    /// Changes the `PartitionStrategy` to always partition any cluster with cardinality greater than the given threshold.
    pub fn with_cardinality_greater_than(self, threshold: usize) -> PartitionStrategy<impl Fn(&Cluster<T, A>) -> bool> {
        self.with_predicate(move |c: &Cluster<T, A>| c.cardinality > threshold)
    }

    /// Changes the `PartitionStrategy` to always partition any cluster with depth less than the given threshold.
    pub fn with_depth_less_than(self, threshold: usize) -> PartitionStrategy<impl Fn(&Cluster<T, A>) -> bool> {
        self.with_predicate(move |c: &Cluster<T, A>| c.depth < threshold)
    }
}

impl<P> PartitionStrategy<P> {
    /// Creates a new `PartitionStrategy` with the given predicate.
    pub fn new(predicate: P) -> Self {
        Self {
            predicate,
            max_split: MaxSplit::None,
            branching_factor: BranchingFactor::default(),
            span_reduction: SpanReductionFactor::default(),
        }
    }

    /// Sets the predicate that determines whether a cluster should be partitioned into child clusters.
    ///
    /// The default predicate will allow partitioning of any cluster with more than one non-center item.
    pub const fn with_predicate<Q>(&self, predicate: Q) -> PartitionStrategy<Q> {
        PartitionStrategy {
            predicate,
            max_split: self.max_split,
            branching_factor: self.branching_factor,
            span_reduction: self.span_reduction,
        }
    }

    /// Sets the maximum size of the largest child cluster when partitioning a cluster.
    ///
    /// If set to `MaxSplit::Fixed` with a value outside the range `[0.5, 1.0)`, it defaults to `MaxSplit::None`.
    pub fn with_max_split(mut self, max_split: MaxSplit) -> Self {
        self.max_split = if let MaxSplit::Fixed(fraction) = max_split {
            if (0.5..1.0).contains(&fraction) {
                MaxSplit::Fixed(fraction)
            } else {
                MaxSplit::None
            }
        } else {
            max_split
        };
        self
    }

    /// Sets the branching factor of the cluster tree.
    ///
    /// The default branching factor is `Unbounded`.
    pub fn with_branching_factor(mut self, branching_factor: BranchingFactor) -> Self {
        self.branching_factor = if let BranchingFactor::Fixed(n) = branching_factor {
            BranchingFactor::Fixed(n.max(2))
        } else if let BranchingFactor::Adaptive(max_k) = branching_factor {
            BranchingFactor::Adaptive(max_k.max(3))
        } else {
            branching_factor
        };
        self
    }

    /// Sets the span reduction factor (SRF).
    ///
    /// The default SRF is `Sqrt2`.
    pub fn with_span_reduction(mut self, span_reduction: SpanReductionFactor) -> Self {
        self.span_reduction = if let SpanReductionFactor::Fixed(srf) = span_reduction {
            srf.into()
        } else {
            span_reduction
        };
        self
    }

    /// Evaluates the predicate on the given cluster. If the predicate returns `true`, the cluster should be partitioned into child clusters.
    pub(crate) fn should_partition<T, A>(&self, cluster: &Cluster<T, A>) -> bool
    where
        P: Fn(&Cluster<T, A>) -> bool,
    {
        (self.predicate)(cluster)
    }

    /// Splits the given items into slices for child clusters according to the partition strategy.
    pub(crate) fn split<'a, Id, I, T, M>(&self, items: &'a mut [(Id, I)], metric: &M, radius_index: usize) -> (T, Splits<'a, Id, I>)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        // Create the first bipolar split.
        let BipolarSplit {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        } = BipolarSplit::new(items, metric, InitialPole::RadialIndex(radius_index));

        let mut splits = if !matches!(self.max_split, MaxSplit::None) {
            self.max_split.split(metric, l_items, r_items, l_distances, r_distances)
        } else if !matches!(self.branching_factor, BranchingFactor::Unbounded) {
            self.branching_factor.split(metric, l_items, r_items, l_distances, r_distances)
        } else {
            self.span_reduction.split(metric, l_items, r_items, l_distances, r_distances, span)
        };
        splits.sort_by_key(|&(i, _)| i);

        (span, splits)
    }
}

impl<P> PartitionStrategy<P>
where
    P: Send + Sync,
{
    /// Parallel version of [`Self::split`].
    pub(crate) fn par_split<'a, Id, I, T, M>(&self, items: &'a mut [(Id, I)], metric: &M, radius_index: usize) -> (T, Splits<'a, Id, I>)
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        // Create the first bipolar split.
        let BipolarSplit {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        } = BipolarSplit::par_new(items, metric, InitialPole::RadialIndex(radius_index));

        let mut splits = if !matches!(self.max_split, MaxSplit::None) {
            self.max_split.par_split(metric, l_items, r_items, l_distances, r_distances)
        } else if !matches!(self.branching_factor, BranchingFactor::Unbounded) {
            self.branching_factor.par_split(metric, l_items, r_items, l_distances, r_distances)
        } else {
            self.span_reduction.par_split(metric, l_items, r_items, l_distances, r_distances, span)
        };
        splits.sort_by_key(|&(i, _)| i);

        (span, splits)
    }
}
