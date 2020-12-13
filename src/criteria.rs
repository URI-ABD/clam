use crate::cluster::Cluster;
use crate::metric::Number;

// TODO: Enum for criteria because you currently cannot have vec![MaxDepth, MinPoints].

// pub trait ClusterCriterion: Send + Sync {
//     fn check<T: Real, U: Real>(&self, cluster: &Cluster<T, U>) -> bool;
// }
//
// #[derive(Debug)]
// pub struct MaxDepth { depth: usize }
//
// impl MaxDepth {
//     pub fn new(depth: usize) -> Arc<Self> {
//         Arc::new(MaxDepth { depth })
//     }
// }
//
// impl ClusterCriterion for MaxDepth {
//     fn check<T: Real, U: Real>(&self, cluster: &Cluster<T, U>) -> bool {
//         cluster.depth() < self.depth
//     }
// }

// TODO: Investigate returning a closure to make criteria.
//  Problem: complains of opaque trait usage when you put different functions together

// TODO: Implement Send and Sync for these closures

// pub trait ClusterCriterion<T, U>: (Fn(&Cluster<T, U>) -> bool) {}

pub fn max_depth<T: Real, U: Real>(depth: usize) -> impl Fn(&Cluster<T, U>) -> bool {
    move |cluster| cluster.depth() < depth
}

pub fn min_points<T: Real, U: Real>(points: usize) -> impl Fn(&Cluster<T, U>) -> bool {
    move |cluster| cluster.cardinality() > points
}

// I'm sorry for this...
pub fn compose<T: Real, U: Real>(
    left: impl Fn(&Cluster<T, U>) -> bool,
    right: impl Fn(&Cluster<T, U>) -> bool,
) -> impl Fn(&Cluster<T, U>) -> bool {
    move |cluster| left(cluster) && right(cluster)
}

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
    use super::max_depth;

    #[test]
    fn test_max_depth() {
        let _criterion = max_depth::<f32, f64>(5);
        // assert_eq!(format!("{:?}", criterion), "MaxDepth { depth: 5 }");
    }
}
