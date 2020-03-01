extern crate rand;

use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use rand::seq;
use rand::seq::SliceRandom;

use super::criteria::Criterion;
use super::dataset::Dataset;
use super::types::*;

type Children<T> = Option<Vec<Rc<Cluster<T>>>>;

const K: u8 = 2;
const SUBSAMPLE: usize = 100;

#[derive(Debug)]
pub struct Cluster<T> {
    pub dataset: Rc<Dataset<T>>,
    pub indices: Indices,
    pub name: String,
    pub children: Children<T>,
}

impl<T> PartialEq for Cluster<T> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.indices == other.indices
    }
}

impl<T> Eq for Cluster<T> {}

impl<T> Hash for Cluster<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl<T> fmt::Display for Cluster<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<T> Cluster<T> {
    pub fn new(dataset: Rc<Dataset<T>>, indices: Indices) -> Cluster<T> {
        Cluster::<T> {
            dataset,
            indices,
            name: String::from(""),
            children: None,
        }
    }

    pub fn cardinality(&self) -> usize {
        self.indices.len()
    }

    pub fn depth(&self) -> usize {
        self.name.len()
    }

    pub fn cluster_count(&self) -> usize {
        match self.children.as_ref() {
            Some(c) => c.iter().map(|c| c.cluster_count()).sum::<usize>() + 1,
            None => 1,
        }
    }

    pub fn argsamples(&self) -> Indices {
        if self.cardinality() <= SUBSAMPLE {
            self.indices.clone()
        } else {
            let n = (self.cardinality() as f64).sqrt() as usize;
            let mut rng = &mut rand::thread_rng();
            self.indices.choose_multiple(&mut rng, n).cloned().collect()
        }
    }

    pub fn partition(self, criteria: &Vec<impl Criterion<T>>) -> Cluster<T> {
        for criterion in criteria.iter() {
            if criterion.check(&self) == false {
                return self;
            }
        }

        let mut argmax: Index = self.argsamples()[0];
        let mut max: f64 = 0.;
        // Get two poles
        let farthest = argmax;

        // Group points
        // Sort groups by cardinality
        let mut children = Vec::new();
        for i in 0..K {
            let c = Cluster::<T> {
                dataset: Rc::clone(&self.dataset),
                indices: vec![0], // TODO
                name: format!("{}{}", self.name, i),
                children: None,
            };
            children.push(Rc::new(c));
        }

        Cluster::<T> {
            dataset: self.dataset,
            indices: self.indices,
            name: self.name,
            children: Some(children),
        }
    }

    pub fn leaves(&self, depth: usize) -> Vec<&Cluster<T>> {
        if self.depth() == depth {
            vec![self]
        } else {
            match self.children.as_ref() {
                Some(c) => c.iter().flat_map(|c| c.leaves(depth)).collect(),
                None => vec![self],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn dataset() -> Rc<Dataset<u64>> {
        let data = Data::<u64>::zeros((2, 2));
        Rc::new(Dataset::new(data, "euclidean"))
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    #[test]
    fn test_hash_eq() {
        let a = Cluster::new(dataset(), vec![0, 1]);
        let b = Cluster::new(dataset(), vec![0, 1]);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }

    #[test]
    fn test_cardinality() {
        let c = Cluster::new(dataset(), vec![0, 1]);
        assert_eq!(c.cardinality(), 2);
        let c = Cluster::new(dataset(), vec![0]);
        assert_eq!(c.cardinality(), 1);
    }

    #[test]
    fn test_display() {
        let c = Cluster::new(dataset(), vec![0, 1]);
        let s = format!("{}", c);
        assert_eq!(s, String::from(""));
    }

    #[test]
    fn test_depth() {
        let c = Cluster::new(dataset(), vec![0, 1]);
        assert_eq!(c.depth(), 0);
        let c = Cluster::<u64> {
            dataset: dataset(),
            indices: vec![0, 1],
            name: String::from("010"),
            children: None,
        };
        assert_eq!(c.depth(), 3);
    }
}
