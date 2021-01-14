use std::marker::{Send, Sync};

use crate::cluster::Cluster;
use crate::metric::Number;

// TODO: Enum for criteria because you currently cannot have vec![MaxDepth, MinPoints].

pub trait ClusterCriterion: Send + Sync {
    fn check<T: Number, U: Number>(&self, cluster: &Cluster<T, U>) -> bool;
}

#[derive(Debug)]
pub struct MaxDepth { depth: usize }

impl MaxDepth {
    pub fn new(depth: usize) -> Box<Self> {
        Box::new(MaxDepth { depth })
    }
}

impl ClusterCriterion for MaxDepth {
    fn check<T: Number, U: Number>(&self, cluster: &Cluster<T, U>) -> bool {
        cluster.depth() < self.depth
    }
}

#[derive(Debug)]
pub struct MinPoints { points: usize }

impl MinPoints {
    pub fn new(points: usize) -> Box<Self> {
        Box::new(MinPoints { points })
    }
}

impl ClusterCriterion for MinPoints {
    fn check<T: Number, U: Number>(&self, cluster: &Cluster<T, U>) -> bool {
        cluster.cardinality() > self.points
    }
}

#[cfg(test)]
mod tests {
    use super::MaxDepth;

    #[test]
    fn test_max_depth() {
        let criterion = MaxDepth::new(5);
        assert_eq!(format!("{:?}", criterion), "MaxDepth { depth: 5 }");
    }
}
