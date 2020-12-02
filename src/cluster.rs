use std::{fmt, sync::Arc};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use dashmap::DashSet;
use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::criteria::ClusterCriterion;
use crate::dataset::Dataset;
use crate::types::{Index, Indices};
use crate::utils::{argmax, argmin};

const SUB_SAMPLE: usize = 100;

#[derive(Debug)]
pub struct Cluster {
    pub dataset: Arc<Dataset>,
    pub name: String,
    pub indices: Indices,
    pub children: Option<(Arc<Cluster>, Arc<Cluster>)>,
    argsamples: Indices,
    pub argcenter: Index,
    pub argradius: Index,
    pub radius: f64,
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
            argsamples: vec![],
            argcenter: 0,
            argradius: 0,
            radius: 0.,
        };
        cluster.argsamples = cluster.argsamples();
        cluster.argcenter = cluster.argcenter();
        cluster.argradius = cluster.argradius();
        cluster.radius = cluster.radius();
        cluster
    }

    pub fn cardinality(&self) -> usize { self.indices.len() }

    pub fn depth(&self) -> usize { self.name.len() }

    pub fn contains(&self, i: &Index) -> bool { self.indices.contains(i) }

    pub fn center(&self) -> ArrayView1<f64> { self.dataset.row(self.argcenter) }

    pub fn is_singleton(&self) -> bool { self.nsamples() == 1 }

    fn poles(&self) -> (Index, Index) {
        if self.nsamples() > 2 {
            let indices = self.indices
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

    #[allow(clippy::ptr_arg, clippy::suspicious_map)]
    pub fn partition(self, criteria: &Vec<Arc<impl ClusterCriterion>>) -> Cluster {
        // TODO: Think about making this non-recursive and making returning children instead.
        //       This would let us extract layer-graph easier.
        //  Problem: parent needs ref to children, AND
        //           outer function to handle parallel partitions needs MUTABLE ref to children.
        if self.is_singleton() { self }
        else {
            for criterion in criteria.iter() {
                if !criterion.check(&self) { return self; }
            }

            let (left, right) = self.poles();
            let indices = self.indices
                .par_iter()
                .filter(|&&i| i != left && i != right)
                .cloned()
                .collect();
            let left_distances = self.dataset.distances_from(left, &indices);
            let right_distances = self.dataset.distances_from(right, &indices);
            let (left_indices, right_indices) = (DashSet::new(), DashSet::new());
            left_indices.insert(left);
            right_indices.insert(right);

            indices.par_iter()
                .zip(left_distances.par_iter()
                         .zip(right_distances.par_iter()))
                .map(|(&i, (&l, r))| {
                    match l.partial_cmp(r).unwrap() {
                        Ordering::Less => left_indices.insert(i),
                        _ => right_indices.insert(i),
                    }})
                .count();

            let (left_indices, right_indices) = if right_indices.len() < left_indices.len()
            { (right_indices, left_indices) }
            else { (left_indices, right_indices) };

            let (left, right) = rayon::join(
                || Cluster::new(
                    Arc::clone(&self.dataset),
                    format!("{}{}", self.name, 0),
                    left_indices.into_iter().collect()
                ).partition(criteria),
                || Cluster::new(
                    Arc::clone(&self.dataset),
                    format!("{}{}", self.name, 1),
                    right_indices.into_iter().collect()
                ).partition(criteria),
            );

            Cluster {
                dataset: self.dataset,
                name: self.name,
                indices: self.indices,
                children: Some((Arc::new(left), Arc::new(right))),
                argsamples: self.argsamples,
                argcenter: self.argcenter,
                argradius: self.argradius,
                radius: self.radius
            }
        }
    }

    pub fn num_descendents(&self) -> usize {
        match self.children.borrow() {
            Some((left, right)) => {
                left.num_descendents() + right.num_descendents() + 2
            },
            None => 0,
        }
    }

    fn argsamples(&self) -> Indices {
        if self.cardinality() <= SUB_SAMPLE { self.indices.clone() }
        else {
            let n = (self.cardinality() as f64).sqrt() as Index;
            self.dataset.choose_unique(self.indices.clone(), n)
        }
    }

    fn nsamples(&self) -> Index { self.argsamples.len() }

    fn argcenter(&self) -> Index {
        let distances = self.dataset.pairwise_distances(&self.argsamples)
            .par_iter()
            .map(|v| v.par_iter().sum())
            .collect();
        let (argcenter, _) = argmin(&distances);
        self.argsamples[argcenter]
    }

    fn argradius(&self) -> Index {
        let distances = self.dataset.distances_from(self.argcenter, &self.indices);
        let (argradius, _) = argmax(&distances);
        self.indices[argradius]
    }

    fn radius(&self) -> f64 { self.dataset.distance(self.argcenter, self.argradius) }
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
        let dataset = Dataset::new(data, "euclidean", false);
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
        ].join(" ");
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
