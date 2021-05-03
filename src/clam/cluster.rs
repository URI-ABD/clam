use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dashmap::DashSet;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{argmax, argmin};

const SUB_SAMPLE: usize = 100;
type Children<T, U> = (Arc<Cluster<T, U>>, Arc<Cluster<T, U>>);

#[derive(Debug)]
pub struct Cluster<T: Number, U: Number> {
    pub dataset: Arc<dyn Dataset<T, U>>,
    pub name: String,
    pub indices: Indices,
    pub children: Option<Children<T, U>>,
    argsamples: Indices,
    pub argcenter: Index,
    pub argradius: Index,
    pub radius: U,
}

impl<T: Number, U: Number> PartialEq for Cluster<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<T: Number, U: Number> Eq for Cluster<T, U> {}

impl<T: Number, U: Number> Hash for Cluster<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl<T: Number, U: Number> fmt::Display for Cluster<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl<T: Number, U: Number> Cluster<T, U> {
    pub fn new(dataset: Arc<dyn Dataset<T, U>>, name: String, indices: Indices) -> Cluster<T, U> {
        let mut cluster = Cluster {
            dataset,
            name,
            indices,
            children: None,
            argsamples: vec![],
            argcenter: 0,
            argradius: 0,
            radius: U::zero(),
        };
        cluster.argsamples = cluster.argsamples();
        cluster.argcenter = cluster.argcenter();
        cluster.argradius = cluster.argradius();
        cluster.radius = cluster.radius();
        cluster
    }

    pub fn cardinality(&self) -> Index {
        self.indices.len()
    }

    pub fn depth(&self) -> usize {
        self.name.len()
    }

    pub fn contains(&self, i: &Index) -> bool {
        self.indices.contains(i)
    }

    pub fn center(&self) -> Arc<ArrayView<T, IxDyn>> {
        self.dataset.instance(self.argcenter)
    }

    pub fn is_singleton(&self) -> bool {
        self.nsamples() == 1
    }

    pub fn distance_to(&self, other: &Arc<Cluster<T, U>>) -> U {
        self.dataset.distance(self.argcenter, other.argcenter)
    }

    #[allow(clippy::ptr_arg)]
    pub fn descend_towards(&self, cluster: &String) -> Result<Arc<Cluster<T, U>>, String> {
        match self.children.borrow() {
            Some((left, right)) => {
                if left.name == cluster[0..left.depth()] {
                    Ok(Arc::clone(left))
                } else if right.name == cluster[0..right.depth()] {
                    Ok(Arc::clone(right))
                } else {
                    Err(format!("Cluster {:} not found.", cluster))
                }
            }
            None => Err(format!("Cluster {:} not found.", cluster)),
        }
    }

    fn poles(&self) -> (Index, Index) {
        if self.nsamples() > 2 {
            let indices = self
                .indices
                .par_iter()
                .filter(|&&i| i != self.argradius)
                .cloned()
                .collect();
            let distances = self.dataset.distances_from(self.argradius, &indices);
            let (farthest, _) = argmax(&distances);
            (self.argradius, indices[farthest])
        } else {
            (self.argsamples[0], self.argsamples[1])
        }
    }

    #[allow(clippy::ptr_arg)]
    pub fn partition(self, criteria: &Vec<Box<impl criteria::ClusterCriterion>>) -> Cluster<T, U> {
        // TODO: Think about making this non-recursive and making returning children instead.
        //       This would let us extract layer-graph easier.
        //  Problem: parent needs ref to children, AND
        //           outer function to handle parallel partitions needs MUTABLE ref to children.
        if self.is_singleton() {
            self
        } else {
            for criterion in criteria.iter() {
                if !criterion.check(&self) {
                    return self;
                }
            }

            let (left, right) = self.poles();
            let indices = self
                .indices
                .par_iter()
                .filter(|&&i| i != left && i != right)
                .cloned()
                .collect();
            let left_distances = self.dataset.distances_from(left, &indices);
            let right_distances = self.dataset.distances_from(right, &indices);
            let (left_indices, right_indices) = (DashSet::new(), DashSet::new());
            left_indices.insert(left);
            right_indices.insert(right);

            indices
                .par_iter()
                .zip(left_distances.par_iter().zip(right_distances.par_iter()))
                .for_each(|(&i, (&l, &r))| {
                    if l <= r {
                        left_indices.insert(i);
                    } else {
                        right_indices.insert(i);
                    }
                });

            let (left_indices, right_indices) = if right_indices.len() < left_indices.len() {
                (right_indices, left_indices)
            } else {
                (left_indices, right_indices)
            };

            let (left, right) = rayon::join(
                || {
                    Cluster::new(
                        Arc::clone(&self.dataset),
                        format!("{}{}", self.name, 0),
                        left_indices.into_iter().collect(),
                    )
                    .partition(criteria)
                },
                || {
                    Cluster::new(
                        Arc::clone(&self.dataset),
                        format!("{}{}", self.name, 1),
                        right_indices.into_iter().collect(),
                    )
                    .partition(criteria)
                },
            );

            Cluster {
                dataset: self.dataset,
                name: self.name,
                indices: self.indices,
                children: Some((Arc::new(left), Arc::new(right))),
                argsamples: self.argsamples,
                argcenter: self.argcenter,
                argradius: self.argradius,
                radius: self.radius,
            }
        }
    }

    pub fn num_descendents(&self) -> usize {
        match self.children.borrow() {
            Some((left, right)) => left.num_descendents() + right.num_descendents() + 2,
            None => 0,
        }
    }

    fn argsamples(&self) -> Indices {
        if self.cardinality() <= SUB_SAMPLE {
            self.indices.clone()
        } else {
            let n = (self.cardinality() as f64).sqrt() as Index;
            self.dataset.choose_unique(self.indices.clone(), n)
        }
    }

    fn nsamples(&self) -> Index {
        self.argsamples.len()
    }

    fn argcenter(&self) -> Index {
        let distances: Vec<U> = self
            .dataset
            .pairwise_distances(&self.argsamples)
            .par_iter()
            .map(|v| v.par_iter().cloned().sum())
            .collect();
        let (argcenter, _) = argmin(&distances);
        self.argsamples[argcenter]
    }

    fn argradius(&self) -> Index {
        let distances = self.dataset.distances_from(self.argcenter, &self.indices);
        let (argradius, _) = argmax(&distances);
        self.indices[argradius]
    }

    fn radius(&self) -> U {
        self.dataset.distance(self.argcenter, self.argradius)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{arr2, Array2};

    use crate::prelude::*;
    use crate::dataset::RowMajor;

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = arr2(&[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let dataset: Arc<dyn Dataset<f64, f64>> =
            Arc::new(RowMajor::<f64, f64>::new(data, "euclidean", false).unwrap());
        let mut criteria = Vec::new();
        criteria.push(criteria::MaxDepth::new(3));
        // criteria.push(MinPoints::new(10));
        let cluster = Cluster::new(
            Arc::clone(&dataset),
            String::from(""),
            dataset.indices().clone(),
        )
        .partition(&criteria);

        assert_eq!(cluster, cluster);
        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 4);
        assert_eq!(cluster.num_descendents(), 6);
        assert!(cluster.radius > 0.);
        assert!(cluster.contains(&0));
        assert!(cluster.contains(&1));
        assert!(cluster.contains(&2));
        assert!(cluster.contains(&3));

        assert_eq!(format!("{:}", cluster), "");
        let cluster_str = vec![
            "Cluster {".to_string(),
            format!("dataset: {:?},", cluster.dataset),
            format!("name: {:?},", cluster.name),
            format!("indices: {:?},", cluster.indices),
            format!("children: {:?},", cluster.children),
            format!("argsamples: {:?},", cluster.argsamples),
            format!("argcenter: {:?},", cluster.argcenter),
            format!("argradius: {:?},", cluster.argradius),
            format!("radius: {:?}", cluster.radius),
            "}".to_string(),
        ]
        .join(" ");
        assert_eq!(format!("{:?}", cluster), cluster_str);

        let (left, right) = cluster.children.unwrap();
        assert_eq!(format!("{:}", left), "0");
        assert_eq!(format!("{:}", right), "1");

        for child in [left, right].iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 2);
            assert_eq!(child.num_descendents(), 2);
        }
    }
}
