//! Criteria used for partitioning `Cluster`s.

use distances::Number;

use crate::Cluster;

/// A criterion used to decide when to partition a `Cluster`.
///
/// # Type Parameters
///
/// - `T`: The type of the instances in the `Cluster`.
/// - `U`: The type of the distance values between instances.
pub trait PartitionCriterion<T: Send + Sync + Copy, U: Number>: Send + Sync {
    /// Check whether a `Cluster` meets the criterion for partitioning.
    // TODO: Figure out how to have this not leak `Cluster` or make `Cluster` public.
    fn check(&self, c: &Cluster<T, U>) -> bool;
}

/// A collection of criteria used to decide when to partition a `Cluster`.
#[allow(clippy::module_name_repetitions)]
pub struct PartitionCriteria<T: Send + Sync + Copy, U: Number> {
    /// The criteria used to decide when to partition a `Cluster`.
    criteria: Vec<Box<dyn PartitionCriterion<T, U>>>,
    /// Whether all criteria must be met for a `Cluster` to be partitioned or if any one criterion
    /// is sufficient.
    check_all: bool,
}

impl<T: Send + Sync + Copy, U: Number> Default for PartitionCriteria<T, U> {
    fn default() -> Self {
        Self::new(true).with_min_cardinality(1)
    }
}

impl<T: Send + Sync + Copy, U: Number> PartitionCriteria<T, U> {
    /// Create a new `PartitionCriteria` instance.
    ///
    /// # Arguments
    ///
    /// * `check_all`: if `true`, all criteria must be met for a `Cluster` to be partitioned, if
    /// `false`, any one criterion is sufficient.
    #[must_use]
    pub fn new(check_all: bool) -> Self {
        Self {
            criteria: Vec::new(),
            check_all,
        }
    }

    /// Add the `MaxDepth` criterion to the collection of criteria.
    ///
    /// # Arguments
    ///
    /// * `threshold`: the maximum depth of a `Cluster` beyond which it may not be partitioned.
    #[must_use]
    pub fn with_max_depth(mut self, threshold: usize) -> Self {
        self.criteria.push(Box::new(MaxDepth(threshold)));
        self
    }

    /// Add the `MinCardinality` criterion to the collection of criteria.
    ///
    /// # Arguments
    ///
    /// * `threshold`: the minimum cardinality of a `Cluster` below which it may not be partitioned.
    #[must_use]
    pub fn with_min_cardinality(mut self, threshold: usize) -> Self {
        self.criteria.push(Box::new(MinCardinality(threshold)));
        self
    }

    /// Add a custom criterion to the collection of criteria.
    ///
    /// # Arguments
    ///
    /// * `c`: the custom criterion to add.
    #[allow(dead_code)]
    pub(crate) fn with_custom(mut self, c: Box<dyn PartitionCriterion<T, U>>) -> Self {
        self.criteria.push(c);
        self
    }
}

impl<T: Send + Sync + Copy, U: Number> PartitionCriterion<T, U> for PartitionCriteria<T, U> {
    fn check(&self, cluster: &Cluster<T, U>) -> bool {
        !cluster.is_singleton()
            && if self.check_all {
                self.criteria.iter().all(|c| c.check(cluster))
            } else {
                self.criteria.iter().any(|c| c.check(cluster))
            }
    }
}

/// The maximum depth of a `Cluster` beyond which it may not be partitioned.
#[derive(Debug, Clone)]
pub struct MaxDepth(usize);

impl<T: Send + Sync + Copy, U: Number> PartitionCriterion<T, U> for MaxDepth {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.depth() < self.0
    }
}

/// The minimum cardinality of a `Cluster` below which it may not be partitioned.
#[derive(Debug, Clone)]
pub struct MinCardinality(usize);

impl<T: Send + Sync + Copy, U: Number> PartitionCriterion<T, U> for MinCardinality {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.cardinality > self.0
    }
}
