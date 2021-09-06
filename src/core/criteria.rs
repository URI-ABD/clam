//! Criteria used for partitioning `Clusters` and selecting `Clusters` for `Graphs`.

use std::sync::Arc;

use crate::core::Ratios;
use crate::prelude::*;

/// A `Box`ed function that decides whether a `Cluster` can be partitioned.
pub type PartitionCriterion<T, U> =
    Box<dyn (Fn(&Arc<Cluster<T, U>>) -> bool) + Send + Sync>;

/// A `Cluster` must have a `depth` lower than the given threshold for it to be partitioned.
pub fn max_depth<T: Number, U: Number>(
    threshold: usize,
) -> PartitionCriterion<T, U> {
    Box::new(move |cluster: &Arc<Cluster<T, U>>| cluster.depth() < threshold)
}

/// A `Cluster` must have a `cardinality` higher than the given threshold for it to be partitioned.
pub fn min_cardinality<T: Number, U: Number>(
    threshold: usize,
) -> PartitionCriterion<T, U> {
    Box::new(move |cluster: &Arc<Cluster<T, U>>| cluster.cardinality > threshold)
}

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<dyn (Fn(Ratios) -> f64) + Send + Sync>;
