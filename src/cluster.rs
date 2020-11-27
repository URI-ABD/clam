use std::{fmt, sync::Arc};
use std::hash::{Hash, Hasher};

use ndarray::{Array1, ArrayView1, Axis};
use rand::prelude::*;

use super::criteria::Criterion;
use super::dataset::Dataset;
use super::types::*;

type Children = Vec<Arc<Cluster>>;

const SUB_SAMPLE: usize = 100;

#[derive(Debug)]
pub struct Cluster {
    dataset: Arc<Dataset>,
    pub name: String,
    pub indices: Indices,
    pub children: Option<Children>,
    argcenter: Option<Index>,
    argradius: Option<Index>,
    radius: Option<f64>,
}

impl PartialEq for Cluster {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for Cluster {}

impl Hash for Cluster {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl Cluster {
    pub fn new(dataset: Arc<Dataset>, name: String, indices: Indices) -> Cluster {
        let mut cluster = Cluster {
            dataset,
            name,
            indices,
            children: None,
            argcenter: None,
            argradius: None,
            radius: None,
        };
        cluster.argcenter = Some(cluster.argcenter());
        cluster.argradius = Some(cluster.argradius());
        cluster.radius = Some(cluster.radius());
        cluster
    }

    pub fn cardinality(&self) -> usize {
        self.indices.len()
    }

    pub fn depth(&self) -> usize {
        self.name.len()
    }

    pub fn contains(&self, i: &Index) -> bool {
        self.indices.contains(i)
    }

    fn argsamples(&self) -> Indices {
        // TODO: cache
        if self.cardinality() <= SUB_SAMPLE {
            self.indices.clone()
        } else {
            let n = (self.cardinality() as f64).sqrt() as Index;
            let mut rng = &mut rand::thread_rng();
            // TODO: check for uniqueness among samples, perhaps in dataset struct
            self.indices.choose_multiple(&mut rng, n).cloned().collect()
        }
    }

    fn nsamples(&self) -> Index { self.argsamples().len() }

    pub fn argcenter(&self) -> Index {
        match self.argcenter {
            Some(argcenter) => argcenter,
            None => {
                let argsamples = self.argsamples();
                let distances: Array1<f64> = self
                    .dataset
                    .distances_among(&argsamples, &argsamples)
                    .sum_axis(Axis(1));

                let mut argcenter = 0;
                for (i, &value) in distances.iter().enumerate() {
                    if value < distances[argcenter] { argcenter = i; }
                }
                argsamples[argcenter]
            }
        }
    }

    pub fn center(&self) -> ArrayView1<f64> { self.dataset.row(self.argcenter()) }

    pub fn argradius(&self) -> Index {
        match self.argradius {
            Some(argradius) => argradius,
            None => {
                let argsamples = self.argsamples();
                let distances = self
                    .dataset
                    .distances_from(self.argcenter(), &argsamples);

                let mut argradius = 0;
                for (i, &value) in distances.iter().enumerate() {
                    if value > distances[argradius] { argradius = i; }
                }
                argsamples[argradius]
            }
        }
    }

    pub fn radius(&self) -> f64 {
        match self.radius {
            Some(radius) => radius,
            None => self.dataset.distance(self.argcenter(), self.argradius()),
        }
    }

    fn poles(&self) -> (Index, Index) {
        let argsamples = self.argsamples();
        if argsamples.len() > 2 {
            let argradius = self.argradius();
            let distances = self.dataset.distances_from(argradius, &argsamples);

            let mut farthest = 0;
            for (i, &value) in distances.iter().enumerate() {
                if value > distances[farthest] { farthest = i; }
            }
            (argradius, argsamples[farthest])
        } else {
            (argsamples[0], argsamples[1])
        }
    }

    #[allow(clippy::ptr_arg)]
    pub fn partition(self, criteria: &Vec<impl Criterion>) -> Cluster {
        // TODO: Think about making this non-recursive and making returning children instead. This would let us extract layer-graph easier.

        if self.nsamples() == 1 {
            self
        } else {
            for criterion in criteria.iter() {
                if !criterion.check(&self) { return self; }
            }

            let (left, right) = self.poles();
            let left_distances = self.dataset.distances_from(left, &self.indices);
            let right_distances = self.dataset.distances_from(right, &self.indices);
            let (mut left_indices, mut right_indices) = (vec![], vec![]);

            for (i, (&l, &r)) in left_distances
                .iter()
                .zip(right_distances.iter())
                .enumerate()
            {
                if l < r {
                    left_indices.push(self.indices[i]);
                } else {
                    right_indices.push(self.indices[i]);
                }
            }

            assert!(!left_indices.is_empty());
            assert!(!right_indices.is_empty());

            let left = Cluster::new(
                Arc::clone(&self.dataset),
                format!("{}{}", self.name, 0),
                left_indices
            ).partition(criteria);
            let right = Cluster::new(
                Arc::clone(&self.dataset),
                format!("{}{}", self.name, 1),
                right_indices
            ).partition(criteria);

            Cluster {
                dataset: self.dataset,
                name: self.name,
                indices: self.indices,
                children: Some(vec![Arc::new(left), Arc::new(right)]),
                argcenter: self.argcenter,
                argradius: self.argradius,
                radius: self.radius
            }
        }
    }

    pub fn num_descendents(&self) -> usize {
        match self.children.as_ref() {
            Some(children) => {
                children
                    .iter()
                    .map(|child| child.num_descendents())
                    .sum::<usize>()
                    + children.len()
            }
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2, Array2};

    use crate::criteria::MaxDepth;

    use super::*;

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = arr2(&[
            [0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
        ]);
        let dataset = Dataset::new(data, "euclidean");
        let indices = dataset.indices();
        let cluster = Cluster::new(
            Arc::new(dataset),
            String::from(""),
            indices
        ).partition(&vec![MaxDepth::new(3)]);

        assert_eq!(cluster, cluster);
        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendents(), 6);
        assert!(cluster.radius() > 0.);
        assert!(cluster.contains(&0));
        assert!(cluster.contains(&1));
        assert!(cluster.contains(&2));
        assert!(cluster.contains(&3));

        assert_eq!(format!("{:}", cluster), "");
        let cluster_str = format!(
            "dataset: {:?}, name: {:?}, indices: {:?}, children: {:?}, argcenter: {:?}, argradius: {:?}, radius: {:?}",
            cluster.dataset,
            cluster.name,
            cluster.indices,
            cluster.children,
            cluster.argcenter,
            cluster.argradius,
            cluster.radius,
        );
        let cluster_str = ["Cluster { ".to_string(), cluster_str, " }".to_string()].join("");
        assert_eq!(format!("{:?}", cluster), cluster_str);

        let children = cluster.children.unwrap();
        assert_eq!(children.len(), 2);
        for child in children.iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendents(), 2);
        }
        assert_eq!(format!("{:}", children[0]), "0");
        assert_eq!(format!("{:}", children[1]), "1");
    }
}
