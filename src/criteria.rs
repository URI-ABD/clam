//! Criteria used for partitioning `Clusters` and selecting `Clusters` for `Graphs`.

use crate::prelude::*;
/// A `Box` over a `closure` that takes a reference to a `Cluster` and returns a `bool`
/// indicating whether that `Cluster` can be partitioned.
pub type PartitionCriterion<T, U> = Box<dyn (Fn(&Cluster<T, U>) -> bool) + Send + Sync>;

/// A `Cluster` must have a `depth` lower than the given threshold for it to be partitioned.
pub fn max_depth<T: Number, U: Number>(threshold: u8) -> PartitionCriterion<T, U> {
    Box::new(move |cluster: &Cluster<T, U>| cluster.depth() < threshold)
}

/// A `Cluster` must have a `cardinality` higher than the given threshold for it to be partitioned.
pub fn min_cardinality<T: Number, U: Number>(threshold: usize) -> PartitionCriterion<T, U> {
    Box::new(move |cluster: &Cluster<T, U>| cluster.cardinality > threshold)
}
