use super::criteria::Criterion;
use super::dataset::Dataset;
use super::types::*;

use std::rc::Rc;

type Children = Option<Vec<Rc<Cluster>>>;

const K: u8 = 2;

#[derive(Debug, Eq, PartialEq)]
pub struct Cluster {
    pub dataset: Rc<Dataset>,
    pub indices: Indices,
    pub name: String,
    pub children: Children,
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

    pub fn n(&self) -> u32 {
        let sum: u32 = match self.children.as_ref() {
            Some(c) => c.iter().map(|c| c.n()).sum(),
            None => 0,
        };
        sum + 1
    }

    pub fn partition(self, criteria: &Vec<impl Criterion>) -> Cluster {
        for criteria in criteria.iter() {
            if criteria.check(&self) == false {
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
