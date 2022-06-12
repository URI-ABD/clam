//! Criteria used for partitioning `Clusters` and selecting `Clusters` for `Graphs`.

use super::cluster;
use crate::prelude::*;

pub trait PartitionCriterion<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    fn check(&self, c: &Cluster<T, U>) -> bool;
}

#[derive(Debug)]
pub struct PartitionCriteria<T: Number, U: Number> {
    criteria: Vec<Box<dyn PartitionCriterion<T, U>>>,
    check_all: bool,
}

impl<T: Number, U: Number> PartitionCriteria<T, U> {
    pub fn new(check_all: bool) -> Self {
        Self {
            criteria: Vec::new(),
            check_all,
        }
    }

    pub fn with_max_depth(mut self, threshold: usize) -> Self {
        self.criteria.push(Box::new(MaxDepth(threshold)));
        self
    }

    pub fn with_min_cardinality(mut self, threshold: usize) -> Self {
        self.criteria.push(Box::new(MinCardinality(threshold)));
        self
    }

    pub fn with_custom(mut self, c: Box<dyn PartitionCriterion<T, U>>) -> Self {
        self.criteria.push(c);
        self
    }

    pub fn check(&self, cluster: &Cluster<T, U>) -> bool {
        !cluster.is_singleton()
            && if self.check_all {
                self.criteria.iter().all(|c| c.check(cluster))
            } else {
                self.criteria.iter().any(|c| c.check(cluster))
            }
    }
}

#[derive(Debug, Clone)]
struct MaxDepth(usize);

impl<T: Number, U: Number> PartitionCriterion<T, U> for MaxDepth {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.depth() < self.0
    }
}

#[derive(Debug, Clone)]
struct MinCardinality(usize);

impl<T: Number, U: Number> PartitionCriterion<T, U> for MinCardinality {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.cardinality() > self.0
    }
}

// A `Box`ed function that decides whether a `Cluster` can be partitioned.

// A `Cluster` must have a `depth` lower than the given threshold for it to be partitioned.
// pub fn max_depth<T: Number, U: Number>(threshold: usize) -> PartitionCriterion<T, U> {
//     Box::new(move |cluster: &Cluster<T, U>| cluster.depth() < threshold)
// }

// A `Cluster` must have a `cardinality` higher than the given threshold for it to be partitioned.
// pub fn min_cardinality<T: Number, U: Number>(threshold: usize) -> PartitionCriterion<T, U> {
//     Box::new(move |cluster: &Cluster<T, U>| cluster.cardinality() > threshold)
// }

/// A `Box`ed function that assigns a score for a given `Cluster`.
pub type MetaMLScorer = Box<dyn (Fn(cluster::Ratios) -> f64) + Send + Sync>;
