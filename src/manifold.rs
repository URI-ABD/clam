use std::rc::Rc;
use std::usize;

use petgraph::graph::UnGraph;

use super::cluster::Cluster;
use super::criteria::*;
use super::dataset::Dataset;
use super::types::*;


#[derive(Debug)]
pub struct Manifold<T> {
    pub dataset: Rc<Dataset<T>>,
    pub root: Cluster<T>,
}

impl<T> PartialEq for Manifold<T> {
    fn eq(&self, other: &Self) -> bool {
        self.leaves(None) == other.leaves(None)
    }
}

impl<T> Eq for Manifold<T> {}

impl<T> Manifold<T> {
    pub fn new(data: Box<Data<T>>, metric: Metric, criteria: Vec<impl Criterion<T>>) -> Manifold<T> {
        let d = Dataset { data, metric };
        let d = Rc::new(d);
        Manifold::<T> {
            dataset: Rc::clone(&d),
            root: Cluster::new(Rc::clone(&d), (0..d.len()).collect()).partition(&criteria),
        }
    }

    pub fn cluster_count(&self) -> u32 {
        self.root.cluster_count()
    }

    pub fn graph(&self, _depth: u8) -> UnGraph<Cluster<T>, Radius> {
        panic!()
    }

    pub fn leaves(&self, depth: Option<usize>) -> Vec<&Cluster::<T>> {
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

    fn data() -> Box<Data<u8>> {
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
