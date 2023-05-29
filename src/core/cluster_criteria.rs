//! Criteria used for partitioning `Cluster`s.

use super::cluster::Cluster;
use super::dataset::Dataset;
use super::number::Number;

pub trait PartitionCriterion<T: Number, U: Number, D: Dataset<T, U>>: std::fmt::Debug + Send + Sync {
    fn check(&self, c: &Cluster<T, U, D>) -> bool;
}

#[derive(Debug)]
pub struct PartitionCriteria<T: Number, U: Number, D: Dataset<T, U>> {
    criteria: Vec<Box<dyn PartitionCriterion<T, U, D>>>,
    check_all: bool,
}

impl<'a, T: Number, U: Number, D: Dataset<T, U>> PartitionCriteria<T, U, D> {
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

    pub fn with_custom(mut self, c: Box<dyn PartitionCriterion<T, U, D>>) -> Self {
        self.criteria.push(c);
        self
    }

    pub fn check(&self, cluster: &Cluster<'a, T, U, D>) -> bool {
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

impl<T: Number, U: Number, D: Dataset<T, U>> PartitionCriterion<T, U, D> for MaxDepth {
    fn check(&self, c: &Cluster<T, U, D>) -> bool {
        c.depth() < self.0
    }
}

#[derive(Debug, Clone)]
struct MinCardinality(usize);

impl<T: Number, U: Number, D: Dataset<T, U>> PartitionCriterion<T, U, D> for MinCardinality {
    fn check(&self, c: &Cluster<T, U, D>) -> bool {
        c.cardinality() > self.0
    }
}
