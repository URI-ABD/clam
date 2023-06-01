//! Criteria used for partitioning `Cluster`s.

use super::cluster::Cluster;
use super::number::Number;

// TODO: OWM: This needs to be thought about way more. The issue here is that
// Cluster is getting leaked but I don't know enough about PartitionCriteria to
// say if we need it to be or not
pub(crate) trait PartitionCriterion<U: Number>: std::fmt::Debug + Send + Sync {
    fn check(&self, c: &Cluster<U>) -> bool;
}

#[derive(Debug)]
pub struct PartitionCriteria<U: Number> {
    criteria: Vec<Box<dyn PartitionCriterion<U>>>,
    check_all: bool,
}

impl<U: Number> PartitionCriteria<U> {
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

    #[allow(dead_code)]
    pub(crate) fn with_custom(mut self, c: Box<dyn PartitionCriterion<U>>) -> Self {
        self.criteria.push(c);
        self
    }

    pub(crate) fn check(&self, cluster: &Cluster<U>) -> bool {
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

impl<U: Number> PartitionCriterion<U> for MaxDepth {
    fn check(&self, c: &Cluster<U>) -> bool {
        c.depth() < self.0
    }
}

#[derive(Debug, Clone)]
struct MinCardinality(usize);

impl<U: Number> PartitionCriterion<U> for MinCardinality {
    fn check(&self, c: &Cluster<U>) -> bool {
        c.cardinality() > self.0
    }
}
