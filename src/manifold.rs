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
    pub fn new(
        dataset: Dataset<T>,
        criteria: Vec<impl Criterion<T>>,
    ) -> Manifold<T> {
        let d = Rc::new(dataset);
        Manifold::<T> {
            dataset: Rc::clone(&d),
            root: Cluster::new(Rc::clone(&d), (0..d.len()).collect()).partition(&criteria),
        }
    }

    pub fn cluster_count(&self) -> usize {
        self.root.cluster_count()
    }

    pub fn graph(&self, _depth: u8) -> UnGraph<Cluster<T>, Radius> {
        panic!()
    }

    pub fn leaves(&self, depth: Option<usize>) -> Vec<&Cluster<T>> {
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

    fn new() -> Manifold<u64> {
        let data = Data::from(array![[0, 0], [0, 1]]);
        let metric = "euclidean";
        let dataset = Dataset::new(data, metric);
        let criteria = vec![MinPoints::new(2)];
        Manifold::new(dataset, criteria)
    }

    #[test]
    fn test_new() {
        let manifold = tests::new();
        assert!(manifold.cluster_count() > 0);
    }

    #[test]
    fn test_leaves() {
        let manifold = tests::new();
        assert_eq!(manifold.leaves(Some(0)).len(), 1);
        assert_eq!(manifold.leaves(Some(1)).len(), 2);
    }

    #[test]
    fn test_eq() {
        let a = tests::new();
        let b = tests::new();
        assert_eq!(a, b);
    }
}
