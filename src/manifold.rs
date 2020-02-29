use std::rc::Rc;

use petgraph::graph::UnGraph;

use super::cluster::Cluster;
use super::criteria::*;
use super::dataset::Dataset;
use super::types::*;


#[derive(Debug, Eq)]
pub struct Manifold {
    pub dataset: Rc<Dataset>,
    pub root: Option<Cluster>,
}

impl PartialEq for Manifold {
    fn eq(&self, other: &Self) -> bool {
        self.dataset == other.dataset && self.leaves() == other.leaves()
    }
}

impl Manifold {
    pub fn new(data: Box<Data>, metric: Metric, criteria: Vec<impl Criterion>) -> Manifold {
        let d = Dataset { data, metric };
        let d = Rc::new(d);
        Manifold {
            dataset: Rc::clone(&d),
            root: Some(Cluster::new(Rc::clone(&d), (0..d.len()).collect()).partition(&criteria)),
        }
    }

    pub fn cluster_count(&self) -> u32 {
        self.root.as_ref().unwrap().cluster_count()
    }

    pub fn graph(&self, _depth: u8) -> UnGraph<Cluster, Radius> {
        panic!()
    }

    pub fn leaves(&self) -> Vec<&Cluster> {
        match self.root.as_ref() {
            Some(r) => r.leaves(),
            None => vec![]
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
        assert_ne!(m.root, None);
    }

    #[test]
    fn leaves() {
        let m = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        let l = m.leaves();
        assert_eq!(l.len(), 2);
    }

    #[test]
    fn eq() {
        let a = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        let b = Manifold::new(data(), metric(), vec![MinPoints::new(2)]);
        assert_eq!(a, b);
    }
}
