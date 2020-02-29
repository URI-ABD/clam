use std::rc::Rc;
use std::usize;

use petgraph::graph::UnGraph;

use super::cluster::Cluster;
use super::criteria::*;
use super::dataset::Dataset;
use super::types::*;


#[derive(Debug, Eq)]
pub struct Manifold {
    pub dataset: Rc<Dataset>,
    pub root: Cluster,
}

impl PartialEq for Manifold {
    fn eq(&self, other: &Self) -> bool {
        self.dataset == other.dataset && self.leaves(None) == other.leaves(None)
    }
}

impl Manifold {
    pub fn new(data: Box<Data>, metric: Metric, criteria: Vec<impl Criterion>) -> Manifold {
        let d = Dataset { data, metric };
        let d = Rc::new(d);
        Manifold {
            dataset: Rc::clone(&d),
            root: Cluster::new(Rc::clone(&d), (0..d.len()).collect()).partition(&criteria),
        }
    }

    pub fn cluster_count(&self) -> u32 {
        self.root.cluster_count()
    }

    pub fn graph(&self, _depth: u8) -> UnGraph<Cluster, Radius> {
        panic!()
    }

    pub fn leaves(&self, depth: Option<usize>) -> Vec<&Cluster> {
        match depth {
            Some(d) => self.root.leaves(d),
            None => self.root.leaves(usize::MAX),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn data() -> Box<Data> {
        Box::new(Data::from(array![[0, 0], [0, 1]]))
    }

    fn metric() -> String {
        String::from("euclidean")
    }

    #[test]
    fn new() {
        let data = Data::from(array![[0, 0], [0, 1]]);
        let metric = String::from("euclidean");
        let m = Manifold::new(Box::new(data), metric, vec![MinPoints::new(2)]);
        assert_eq!(m.cluster_count(), 3);
    }

    #[test]
    fn leaves() {
        let m = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        let l = m.leaves(Some(0));
        assert_eq!(l.len(), 1);
        let l = m.leaves(Some(1));
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn eq() {
        let a = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        let b = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        assert_eq!(a, b);
    }
}
