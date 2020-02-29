use super::criteria::Criterion;
use super::dataset::Dataset;
use super::types::*;

use std::rc::Rc;
use std::hash::{Hash, Hasher};

type Children = Option<Vec<Rc<Cluster>>>;

const K: u8 = 2;

#[derive(Debug)]
pub struct Cluster {
    pub dataset: Rc<Dataset>,
    pub indices: Indices,
    pub name: String,
    pub children: Children,
}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.indices == other.indices
    }
}

impl Eq for Cluster {}

impl Hash for Cluster {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Cluster {
    pub fn new(dataset: Rc<Dataset>, indices: Indices) -> Cluster {
        Cluster {
            dataset,
            indices,
            name: String::from(""),
            children: None,
        }
    }

    pub fn cardinality(&self) -> usize {
        self.indices.len()
    }

    pub fn cluster_count(&self) -> u32 {
        match self.children.as_ref() {
            Some(c) => c.iter().map(|c| c.cluster_count()).sum::<u32>() + 1,
            None => 1,
        }
    }

    pub fn partition(self, criteria: &Vec<impl Criterion>) -> Cluster {
        for criterion in criteria.iter() {
            if criterion.check(&self) == false {
                return self;
            }
        }
        let mut children = Vec::new();
        for i in 0..K {
            let c = Cluster {
                dataset: Rc::clone(&self.dataset),
                indices: vec![0],
                name: format!("{}{}", self.name, i),
                children: None,
            };
            children.push(Rc::new(c));
        }

        Cluster {
            dataset: self.dataset,
            indices: self.indices,
            name: self.name,
            children: Some(children),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn dataset() -> Rc<Dataset> {
        Rc::new(Dataset {
            data: Box::new(Data::zeros((2, 2))),
            metric: String::from("euclidean")
        })
    }

    fn hash<T: Hash>(t: &T) -> u64 {
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    #[test]
    fn hash_eq() {
        let a = Cluster::new(dataset(), vec![0, 1]);
        let b = Cluster::new(dataset(), vec![0, 1]);
        assert_eq!(a, b);
        assert_eq!(hash(&a), hash(&b));
    }

    #[test]
    fn cardinality() {
        let c = Cluster::new(dataset(), vec![0, 1]);
        assert_eq!(c.cardinality(), 2);
        let c = Cluster::new(dataset(), vec![0]);
        assert_eq!(c.cardinality(), 1);
    }
}