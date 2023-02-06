//! Criteria used for partitioning `Clusters` and selecting `Clusters` for `Graphs`.

use crate::prelude::*;

pub trait PartitionCriterion<'a, T, S>: std::fmt::Debug + Send + Sync
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    fn check(&self, c: &Cluster<'a, T, S>) -> bool;
}

#[derive(Debug)]
pub struct PartitionCriteria<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    criteria: Vec<Box<dyn PartitionCriterion<'a, T, S>>>,
    check_all: bool,
}

impl<'a, T, S> PartitionCriteria<'a, T, S>
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
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

    pub fn with_custom(mut self, c: Box<dyn PartitionCriterion<'a, T, S>>) -> Self {
        self.criteria.push(c);
        self
    }

    pub fn check(&self, cluster: &Cluster<'a, T, S>) -> bool {
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

impl<'a, T, S> PartitionCriterion<'a, T, S> for MaxDepth
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    fn check(&self, c: &Cluster<'a, T, S>) -> bool {
        c.depth() < self.0
    }
}

#[derive(Debug, Clone)]
struct MinCardinality(usize);

impl<'a, T, S> PartitionCriterion<'a, T, S> for MinCardinality
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    fn check(&self, c: &Cluster<'a, T, S>) -> bool {
        c.cardinality() > self.0
    }
}
