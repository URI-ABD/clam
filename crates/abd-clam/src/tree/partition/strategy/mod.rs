//! How a [`Cluster`](crate::Cluster) is partitioned into child clusters.

use crate::{DistanceValue, NamedAlgorithm};

mod bipolar_split;
mod branching_factor;
mod max_fraction;
mod span_reduction_factor;

pub use branching_factor::BranchingFactor;
pub use max_fraction::MaxFraction;
pub use span_reduction_factor::SpanReductionFactor;

use bipolar_split::{BipolarSplit, InitialPole};

/// A type-alias for the span splits after applying a partition strategy to a slice of items.
pub(crate) type Splits<'a, Id, I> = Vec<(usize, &'a mut [(Id, I)])>;

/// The `PartitionStrategy` determines how a cluster is partitioned into child clusters during tree construction.
///
/// This consists of one or more applications of a partition algorithm, described later, to the non-center items of the cluster. The different variants of
/// `PartitionStrategy` determine how many times the partition algorithm is applied, and how the resulting splits are selected for further partitioning.
///
/// The base partition algorithm is a essentially a bi-polar split of the non-center items. It proceeds as follows:
///
/// 1. Choose an item in the slice and find the item farthest from it. This is the first pole.
/// 2. Find the item farthest from the first pole. This is the second pole.
/// 3. Find the distance from each item to each pole.
/// 4. Assign each item to one of two groups based on which pole it is closer to, breaking ties by assigning to the first pole.
/// 5. Reorder the items in-place so that the items in the first group are contiguous and come before the items in the second group.
/// 6. Split the slice into two sub-slices at the boundary between the two groups.
///
/// Once we decide that a cluster should be partitioned, we apply one pass of this "bipolar split" to the non-center items of the cluster. The two resulting
/// sub-slices are then passed on to the chosen variant of `PartitionStrategy` to determine how to further partition the slices. Different strategies will
/// result in different numbers of child clusters, even for different parents within the same tree. See the documentation for each variant for more details on
/// how they work.
///
/// The default partition strategy, for now, is [`SpanReductionFactor::Sqrt2`]. This was chosen based on theoretical analysis and empirical benchmarks on a
/// variety of datasets, though this may change as we develop more strategies and gather more empirical evidence.
///
/// We provide some helpful predefined strategies, such as [binary](Self::binary), which will always produce exactly two child clusters for any cluster that is
/// partitioned. The goal with these predefined strategies is to help replicate results from our previous publications.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub enum PartitionStrategy {
    /// See [`BranchingFactor`] for more details.
    BranchingFactor(BranchingFactor),
    /// See [`MaxFraction`] for more details.
    MaxFraction(MaxFraction),
    /// See [`SpanReductionFactor`] for more details.
    SpanReductionFactor(SpanReductionFactor),
}

impl core::fmt::Display for PartitionStrategy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BranchingFactor(bf) => bf.fmt(f),
            Self::MaxFraction(mf) => mf.fmt(f),
            Self::SpanReductionFactor(srf) => srf.fmt(f),
        }
    }
}

impl core::str::FromStr for PartitionStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::regex_pattern().captures(s).map_or_else(
            || Err(format!("Invalid format for PartitionStrategy: {s}")),
            |caps| {
                let strategy = caps.get(1).map(|m| m.as_str());
                match strategy {
                    Some("branching-factor") => BranchingFactor::from_str(s).map(Self::from),
                    Some("max-fraction") => MaxFraction::from_str(s).map(Self::from),
                    Some("span-reduction-factor") => SpanReductionFactor::from_str(s).map(Self::from),
                    Some(strategy) => Err(format!("Unknown partition strategy: {strategy}")),
                    None => Err(format!("Invalid format for PartitionStrategy: {s}")),
                }
            },
        )
    }
}

impl NamedAlgorithm for PartitionStrategy {
    fn name(&self) -> &'static str {
        match self {
            Self::BranchingFactor(bf) => bf.name(),
            Self::MaxFraction(mf) => mf.name(),
            Self::SpanReductionFactor(srf) => srf.name(),
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(r"^(branching-factor|max-fraction|span-reduction-factor)(::[a-z0-9\-]+(?:=\d+(?:\.\d+)?)?)?$")
    }
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::SpanReductionFactor(SpanReductionFactor::default())
    }
}

impl PartitionStrategy {
    /// Creates a partition strategy, [`BranchingFactor::Binary`], that will always produce exactly two child clusters for any cluster that is partitioned.
    pub const fn binary() -> Self {
        Self::BranchingFactor(BranchingFactor::Binary)
    }
}

impl PartitionStrategy {
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

        let mut splits = match self {
            Self::BranchingFactor(branching_factor) => branching_factor.split(metric, l_items, r_items, l_distances, r_distances),
            Self::MaxFraction(max_fraction) => max_fraction.split(metric, l_items, r_items, l_distances, r_distances),
            Self::SpanReductionFactor(span_reduction) => span_reduction.split(metric, l_items, r_items, l_distances, r_distances, span),
        };
        splits.sort_by_key(|&(i, _)| i);

        (span, splits)
    }

    /// Parallel version of [`Self::split`].
    pub(crate) fn par_split<'a, Id, I, T, M>(&self, items: &'a mut [(Id, I)], metric: &M, radius_index: usize) -> (T, Splits<'a, Id, I>)
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        profi::prof!("PartitionStrategy::par_split");

        // Create the first bipolar split.
        let BipolarSplit {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        } = BipolarSplit::par_new(items, metric, InitialPole::RadialIndex(radius_index));

        let mut splits = match self {
            Self::BranchingFactor(branching_factor) => branching_factor.par_split(metric, l_items, r_items, l_distances, r_distances),
            Self::MaxFraction(max_fraction) => max_fraction.par_split(metric, l_items, r_items, l_distances, r_distances),
            Self::SpanReductionFactor(span_reduction) => span_reduction.par_split(metric, l_items, r_items, l_distances, r_distances, span),
        };
        splits.sort_by_key(|&(i, _)| i);

        (span, splits)
    }
}
