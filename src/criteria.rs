use super::cluster::Cluster;

pub trait Criterion<T> {
    fn check(&self, cluster: &Cluster<T>) -> bool;
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

impl<T> Criterion<T> for MinPoints {
    fn check(&self, cluster: &Cluster<T>) -> bool {
        cluster.indices.len() > self.points
    }
}
