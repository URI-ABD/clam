use super::cluster::Cluster;

pub trait Criterion {
    fn check(&self, cluster: &Cluster) -> bool;
}

#[derive(Debug)]
pub struct MinPoints {
    points: usize,
}

impl MinPoints {
    pub fn new(points: usize) -> Self {
        MinPoints { points }
    }
}

impl Criterion for MinPoints {
    fn check(&self, cluster: &Cluster) -> bool {
        cluster.indices.len() > self.points
    }
}
