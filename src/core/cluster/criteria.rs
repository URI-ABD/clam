//! Criteria used for partitioning `Cluster`s.

use distances::Number;

use super::Cluster;

// Note (OWM): This leaks cluster if we allow it to be public. Getting this to make sense is a TODO
pub(crate) trait PartitionCriterion<T: Send + Sync + Copy, U: Number>: std::fmt::Debug + Send + Sync {
    // TODO (Najib): figure out how not to lean Cluster here
    fn check(&self, c: &Cluster<T, U>) -> bool;
}

#[derive(Debug)]
pub struct PartitionCriteria<T: Send + Sync + Copy, U: Number> {
    criteria: Vec<Box<dyn PartitionCriterion<T, U>>>,
    check_all: bool,
}

impl<T: Send + Sync + Copy, U: Number> PartitionCriteria<T, U> {
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
    pub(crate) fn with_custom(mut self, c: Box<dyn PartitionCriterion<T, U>>) -> Self {
        self.criteria.push(c);
        self
    }

    pub(crate) fn check(&self, cluster: &Cluster<T, U>) -> bool {
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

impl<T: Send + Sync + Copy, U: Number> PartitionCriterion<T, U> for MaxDepth {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.depth() < self.0
    }
}

#[derive(Debug, Clone)]
struct MinCardinality(usize);

impl<T: Send + Sync + Copy, U: Number> PartitionCriterion<T, U> for MinCardinality {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.cardinality > self.0
    }
}
