use super::cluster::Cluster;

// TODO: Enum for criteria because you currently cannot have vec![MaxDepth, MinPoints].

pub trait Criterion {
    fn check(&self, cluster: &Cluster) -> bool;
}

#[derive(Debug)]
pub struct MaxDepth { depth: usize }

impl MaxDepth {
    pub fn new(depth: usize) -> Self {
        MaxDepth { depth }
    }
}

impl Criterion for MaxDepth {
    fn check(&self, cluster: &Cluster) -> bool {
        cluster.depth() < self.depth
    }
}

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
