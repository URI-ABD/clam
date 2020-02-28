use super::cluster::Cluster;
use super::criteria::*;
use super::dataset::Dataset;
use super::types::*;

use std::rc::Rc;

#[derive(Debug, Eq, PartialEq)]
pub struct Manifold {
    pub data: Rc<Dataset>,
    pub root: Option<Cluster>,
}

impl Manifold {
    pub fn new(data: Box<Data>, metric: Metric, criteria: Vec<impl Criterion>) -> Manifold {
        let d = Dataset { data, metric };
        let d = Rc::new(d);
        Manifold {
            data: Rc::clone(&d),
            root: Some(Cluster::new(Rc::clone(&d), (0..d.len()).collect()).partition(&criteria)),
        }
    }

    pub fn cluster_count(&self) -> u32 {
        self.root.as_ref().unwrap().cluster_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn new() {
        let data = Data::from(array![[0, 0], [0, 1]]);
        let metric = String::from("euclidean");
        let m = Manifold::new(Box::new(data), metric, vec![MinPoints::new(2)]);
        assert_eq!(m.cluster_count(), 3);
        assert_ne!(m.root, None);
    }
}
