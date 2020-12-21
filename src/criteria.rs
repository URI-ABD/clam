use std::marker::{Send, Sync};
use std::sync::Arc;

use crate::cluster::Cluster;
use crate::metric::Number;

// TODO: Enum for criteria because you currently cannot have vec![MaxDepth, MinPoints].

pub trait ClusterCriterion: Send + Sync {
    fn check<T: Number, U: Number>(&self, cluster: &Cluster<T, U>) -> bool;
}

#[derive(Debug)]
pub struct MaxDepth { depth: usize }

impl MaxDepth {
    pub fn new(depth: usize) -> Arc<Self> {
        Arc::new(MaxDepth { depth })
    }
}

impl ClusterCriterion for MaxDepth {
    fn check<T: Number, U: Number>(&self, cluster: &Cluster<T, U>) -> bool {
        cluster.depth() < self.depth
    }
}

// TODO: Investigate returning a closure to make criteria.
//  Problem: complains of opaque trait usage when you put different functions together
// pub fn max_depth<T: Num, U: Num>(depth: usize) -> impl Fn(&Cluster<T, U>) -> bool {
//     move |cluster| cluster.depth() < depth
// }
//
// pub fn min_points<T: Num, U: Num>(points: usize) -> impl Fn(&Cluster<T, U>) -> bool {
//     move |cluster| cluster.cardinality() > points
// }

// #[derive(Debug)]
// pub struct MinPoints { points: usize }
//
// impl MinPoints {
//     pub fn new(points: usize) -> Self { MinPoints { points } }
// }
//
// impl Criterion for MinPoints {
//     fn check(&self, cluster: &Cluster) -> bool { cluster.indices.len() > self.points }
// }

#[cfg(test)]
mod tests {
    use super::MaxDepth;

    #[test]
    fn test_max_depth() {
        let criterion = MaxDepth::new(5);
        assert_eq!(format!("{:?}", criterion), "MaxDepth { depth: 5 }");
    }
}
