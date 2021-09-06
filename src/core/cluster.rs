//! CLAM-Cluster
//!
//! Define and implement the `Cluster` struct.

use std::collections::HashMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::Weak;

use bitvec::prelude::*;
use rayon::prelude::*;

use crate::core::Ratios;
use crate::prelude::*;
use crate::utils::argmax;
use crate::utils::argmin;
use criteria::PartitionCriterion;

const SUB_SAMPLE_LIMIT: usize = 100;

/// A 2-tuple of `Arc<Cluster>` representing the two child `Clusters`
/// formed when a `Cluster` is partitioned.
type Children<T, U> = (Arc<Cluster<T, U>>, Arc<Cluster<T, U>>);
pub type ClusterName = BitVec<Lsb0, u8>;

/// A collection of similar `Instances` from a `Dataset`.
///
/// `Clusters` can be unwieldy to use directly unless you have a
/// good grasp on the underlying invariants.
/// We anticipate that most users' needs will be well met by the higher-level
/// abstractions.
#[derive(Debug)]
pub struct Cluster<T: Number, U: Number> {
    /// An Arc to a struct that implements the Dataset trait.
    pub dataset: Arc<dyn Dataset<T, U>>,

    /// The name of a cluster and is meant to be unique in a tree.
    ///
    /// A Cluster's name is the turn when it would be visited in a breadth-first
    /// traversal of a perfect and balanced binary tree.
    /// The root is named 1 and all descendants follow.
    pub name: ClusterName,

    /// The number of instances in this Cluster.
    pub cardinality: usize,

    /// The `Indices` (a Vec<usize>) of instances in this `Cluster`.
    pub indices: Vec<Index>,

    /// The `Index` of the center of the Cluster.
    pub argcenter: Index,

    /// The `Index` of the instance in the Cluster that is farthest away from the center.
    pub argradius: Index,

    /// The distance from the center to the instance that is farthest away form the center.
    pub radius: U,

    /// The local-fractal dimension of this cluster sampled at its radius and half its radius.
    pub lfd: f64,

    /// Clusters start with no Children and may get Some after partition.
    pub children: RwLock<Option<Children<T, U>>>,

    /// `Weak` arc-ref to the parent `Cluster`. Should be `None` for the root.
    pub parent: Option<Weak<Cluster<T, U>>>,

    /// `Cluster` ratios for meta-ml functions.
    pub ratios: Ratios,
}

impl<T: Number, U: Number> PartialEq for Cluster<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl<T: Number, U: Number> Eq for Cluster<T, U> {}

impl<T: Number, U: Number> PartialOrd for Cluster<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.depth() == other.depth() {
            self.name.partial_cmp(&other.name)
        } else {
            Some(self.depth().cmp(&other.depth()))
        }
    }
}

impl<T: Number, U: Number> Ord for Cluster<T, U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: Number, U: Number> Hash for Cluster<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state)
    }
}

impl<T: Number, U: Number> std::fmt::Display for Cluster<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let name_str: Vec<&str> = self
            .name
            .iter()
            .map(|b| if *b { "1" } else { "0" })
            .collect();
        write!(f, "{}", name_str.join(""))
    }
}

impl<T: Number, U: Number> Cluster<T, U> {
    /// Creates a new root `Cluster` on the entire dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A reference to a struct that implements the `Dataset` trait.
    pub fn new_root(dataset: Arc<dyn Dataset<T, U>>) -> Arc<Self> {
        let name = bitvec![Lsb0, u8; 1];
        let indices = dataset.indices();
        Cluster::new(dataset, name, indices, None, None)
    }

    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A reference to a struct that implements the `Dataset` trait.
    ///   We never copy `Instances` from the `Dataset`.
    /// * `name`: BitVec name for the `Cluster`.
    /// * `indices`: The `Indices` of `Instances` from the dataset that are contained in the `Cluster`.
    /// * `parent`: Weak ref to the parent `Cluster`. Should be `None` for the root.
    pub fn new(
        dataset: Arc<dyn Dataset<T, U>>,
        name: BitVec<Lsb0, u8>,
        indices: Vec<Index>,
        parent: Option<Weak<Cluster<T, U>>>,
        parent_ratios: Option<Ratios>,
    ) -> Arc<Self> {
        let mut cluster = Cluster {
            dataset,
            name,
            cardinality: indices.len(),
            indices,
            children: RwLock::new(None),
            argcenter: 0,
            argradius: 0,
            radius: U::zero(),
            lfd: 0.,
            parent,
            ratios: [1.; 6],
        };
        cluster.argcenter = cluster.argcenter();

        // TODO: Deconstructing assignments require a let-binding.
        //  For now, we need nightly Rust to avoid the let-binding and do this in one line.
        //  When Feature becomes stable, remove this let-binding.
        let (argradius, radius, lfd) = cluster.argradius_radius_lfd();
        cluster.argradius = argradius;
        cluster.radius = radius;
        cluster.lfd = lfd;

        if let Some(parent_ratios) = parent_ratios {
            cluster.ratios = cluster.ratios(parent_ratios);
        }

        Arc::new(cluster)
    }

    /// Returns the ancestors of self in a Vec, excluding self.
    pub fn ancestry(self: &Arc<Self>) -> Vec<Arc<Cluster<T, U>>> {
        let mut ancestry = vec![Arc::clone(self)];
        while ancestry.last().unwrap().depth() > 0 {
            ancestry.push(
                ancestry
                    .last()
                    .unwrap()
                    .parent
                    .clone()
                    .unwrap()
                    .upgrade()
                    .unwrap(),
            );
        }
        ancestry = ancestry.into_iter().rev().collect();
        ancestry.pop();
        ancestry
    }

    fn next_ema(&self, ratio: f64, ema: f64) -> f64 {
        let alpha = 2. / 11.;
        alpha * ratio + (1. - alpha) * ema
    }

    fn ratios(&self, parent_ratios: Ratios) -> Ratios {
        match &self.parent {
            Some(parent) => {
                let parent = parent.upgrade().unwrap();
                let c = (self.cardinality as f64) / (parent.cardinality as f64);
                let r = self.radius.to_f64() / parent.radius.to_f64();
                let l = self.lfd / parent.lfd;

                let [_, _, _, ema_c, ema_r, ema_l] = parent_ratios;
                let ema_c = self.next_ema(c, ema_c);
                let ema_r = self.next_ema(c, ema_r);
                let ema_l = self.next_ema(c, ema_l);

                [c, r, l, ema_c, ema_r, ema_l]
            }
            None => [1.; 6],
        }
    }

    /// The depth of the `Cluster` in the tree. The root `Cluster` has depth 0.
    pub fn depth(&self) -> usize {
        self.name.len() - 1
    }

    /// A reference to the instance which is the (sometimes approximate) geometric median of the `Cluster`.
    ///
    /// TODO: Change the type of `Instance` to something generic.
    /// Ideally, something implementing an `Instance` trait so that `Dataset` becomes a collection of `Instances`.
    pub fn center(&self) -> Vec<T> {
        self.dataset.instance(self.argcenter)
    }

    /// Returns whether this `Cluster` contains only one unique `Instance` or its radius is 0.
    pub fn is_singleton(&self) -> bool {
        self.radius == U::from(0).unwrap()
    }

    pub fn is_ancestor_of(&self, other: &Arc<Cluster<T, U>>) -> bool {
        self.depth() < other.depth()
            && self.name[..] == other.name[..self.name.len()]
    }

    /// Returns the distance from the center of the Cluster to the given instance.
    pub fn distance_to_other(&self, other: &Arc<Cluster<T, U>>) -> U {
        self.dataset.distance(self.argcenter, other.argcenter)
    }

    pub fn distance_to_instance(&self, instance: &[T]) -> U {
        self.dataset.metric().distance(&self.center(), instance)
    }

    /// Find the candidate neighbors for the cluster given the candidates for the parent.
    pub fn find_candidates(
        &self,
        parent_candidates: &HashMap<Arc<Cluster<T, U>>, U>,
    ) -> HashMap<Arc<Cluster<T, U>>, U> {
        let mut candidates: Vec<_> = parent_candidates
            .iter()
            .map(|(c, _)| Arc::clone(c))
            .collect();
        candidates.extend(parent_candidates.iter().flat_map(
            |(candidate, _)| match candidate.children.read().unwrap().clone() {
                Some((left, right)) => vec![left, right],
                None => Vec::new(),
            },
        ));
        candidates
            .par_iter()
            .map(|candidate| {
                (Arc::clone(candidate), self.distance_to_other(candidate))
            })
            .filter(|(candidate, distance)| {
                *distance <= U::from(4).unwrap() * self.radius + candidate.radius
            })
            .collect()
    }

    pub fn add_instance(
        &self,
        sequence: &[T],
        distance: U,
    ) -> HashMap<ClusterName, U> {
        let mut result = match self.children.read().unwrap().clone() {
            Some((left, right)) => {
                let left_distance = left.distance_to_instance(sequence);
                let right_distance = right.distance_to_instance(sequence);

                if left_distance <= right_distance {
                    left.add_instance(sequence, left_distance)
                } else {
                    right.add_instance(sequence, right_distance)
                }
            }
            None => HashMap::new(),
        };
        result.insert(self.name.clone(), distance);
        result
    }

    /// Returns the indices of two maximally separated instances in the Cluster.
    fn poles(&self) -> (Index, Index) {
        let indices: Vec<Index> = self
            .indices
            .par_iter()
            .filter(|&&i| i != self.argradius)
            .cloned()
            .collect();
        let distances = self.dataset.distances_from(self.argradius, &indices);
        let (farthest, _) = argmax(&distances.to_vec());
        (self.argradius, indices[farthest])
    }

    /// Recursively partition the cluster until some `criterion` determines
    /// that a leaf cluster has been reached.
    /// Returns a new cluster containing the built subtree.
    ///
    /// # Arguments
    ///
    /// * `partition_criteria`: A collection of `PartitionCriterion`.
    ///   Each `PartitionCriterion` must evaluate to `true` otherwise the `Cluster`
    ///   cannot be partitioned.
    pub fn partition(
        self: Arc<Self>,
        criteria: &[PartitionCriterion<T, U>],
    ) -> Arc<Self> {
        // Cannot partition a singleton cluster.
        if self.is_singleton() {
            return self;
        }

        // The cluster may only be partitioned if it passes all criteria
        if criteria.par_iter().any(|criterion| !criterion(&self)) {
            return self;
        }

        // Get indices of left and right poles
        let (left, right) = self.poles();

        // Split cluster indices by proximity to left or right pole
        let (left, right): (Vec<Index>, Vec<Index>) =
            self.indices.par_iter().partition(|&&i| {
                self.dataset.distance(left, i) <= self.dataset.distance(i, right)
            });

        // Ensure that left cluster is more populated than right cluster.
        let (left, right) = if right.len() > left.len() {
            (right, left)
        } else {
            (left, right)
        };
        let left_name = {
            let mut name = self.name.clone();
            name.push(false);
            name
        };
        let right_name = {
            let mut name = self.name.clone();
            name.push(true);
            name
        };

        // Recursively apply partition to child clusters.
        let (left, right) = rayon::join(
            || {
                Cluster::new(
                    Arc::clone(&self.dataset),
                    left_name,
                    left,
                    Some(Arc::downgrade(&self)),
                    Some(self.ratios),
                )
                .partition(criteria)
            },
            || {
                Cluster::new(
                    Arc::clone(&self.dataset),
                    right_name,
                    right,
                    Some(Arc::downgrade(&self)),
                    Some(self.ratios),
                )
                .partition(criteria)
            },
        );

        *self.children.write().unwrap() = Some((left, right));
        self
    }

    /// Returns a Vec of clusters containing all descendants of the cluster excluding itself.
    pub fn flatten_tree(&self) -> Vec<Arc<Cluster<T, U>>> {
        match self.children.read().unwrap().clone() {
            Some((left, right)) => {
                let mut descendants =
                    vec![Arc::clone(&left), Arc::clone(&right)];
                descendants.append(&mut left.flatten_tree());
                descendants.append(&mut right.flatten_tree());
                descendants
            }
            None => vec![],
        }
    }

    /// Returns the number of clusters in the subtree rooted at
    /// this cluster (excluding this cluster).
    pub fn num_descendants(&self) -> usize {
        self.flatten_tree().len()
    }

    /// Returns unique samples from among Cluster.indices.
    ///
    /// These significantly speed up the computation of center and partition without much loss in accuracy.
    fn argsamples(&self) -> Vec<Index> {
        if self.cardinality <= SUB_SAMPLE_LIMIT {
            self.indices.clone()
        } else {
            self.dataset.choose_unique(
                self.indices.clone(),
                (self.cardinality as f64).sqrt() as usize,
            )
        }
    }

    /// Returns the `Index` of the `center` of the `Cluster`.
    fn argcenter(&self) -> Index {
        let argsamples = self.argsamples();
        let distances: Vec<U> = self
            .dataset
            .pairwise_distances(&argsamples)
            .outer_iter()
            .map(|v| v.iter().cloned().sum())
            .collect();
        let (argcenter, _) = argmin(&distances);
        argsamples[argcenter]
    }

    /// Returns the index of the farthest point from the center, the distance to that point, and the local fractal dimension of the cluster.
    fn argradius_radius_lfd(&self) -> (Index, U, f64) {
        let distances = self
            .dataset
            .distances_from(self.argcenter, &self.indices)
            .to_vec();
        let (argradius, _) = argmax(&distances);

        let argradius = self.indices[argradius];
        let radius = self.dataset.distance(self.argcenter, argradius);

        let half_count = distances
            .into_par_iter()
            .filter(|&distance| distance <= (radius / U::from(2).unwrap()))
            .count();
        let lfd = if half_count > 0 {
            ((self.cardinality as f64) / (half_count as f64)).log2()
        } else {
            1.
        };

        (argradius, radius, lfd)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::RowMajor;
    use crate::prelude::*;

    #[test]
    fn test_cluster() {
        let data = vec![
            vec![0., 0., 0.],
            vec![1., 1., 1.],
            vec![2., 2., 2.],
            vec![3., 3., 3.],
        ];
        let metric = metric_from_name("euclidean").unwrap();
        let dataset: Arc<dyn Dataset<f64, f64>> =
            Arc::new(RowMajor::<f64, f64>::new(Arc::new(data), metric, false));
        let criteria =
            vec![criteria::max_depth(3), criteria::min_cardinality(1)];
        let cluster =
            Cluster::new_root(Arc::clone(&dataset)).partition(&criteria);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality, 4);
        assert_eq!(cluster.num_descendants(), 6);
        assert!(cluster.radius > 0.);

        assert_eq!(format!("{:}", cluster), "1");
        let cluster_str = vec![
            "Cluster {".to_string(),
            format!("dataset: {:?},", cluster.dataset),
            format!("name: {:?},", cluster.name),
            format!("cardinality: {:?},", cluster.cardinality),
            format!("indices: {:?},", cluster.indices),
            format!("argcenter: {:?},", cluster.argcenter),
            format!("argradius: {:?},", cluster.argradius),
            format!("radius: {:?},", cluster.radius),
            format!("lfd: {:?},", cluster.lfd),
            format!("children: {:?},", cluster.children),
            format!("parent: {:?},", cluster.parent),
            format!("ratios: {:?}", cluster.ratios),
            "}".to_string(),
        ]
        .join(" ");
        assert_eq!(format!("{:?}", cluster), cluster_str);

        let (left, right) = cluster.children.read().unwrap().clone().unwrap();
        assert_eq!(format!("{:}", left), "10");
        assert_eq!(format!("{:}", right), "11");

        for child in [left, right].iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality, 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }

    #[test]
    fn test_ancestry() {
        let data = vec![
            vec![0., 0., 0.],
            vec![1., 1., 1.],
            vec![2., 2., 2.],
            vec![3., 3., 3.],
        ];
        let metric = metric_from_name("euclidean").unwrap();
        let dataset: Arc<dyn Dataset<f64, f64>> =
            Arc::new(RowMajor::<f64, f64>::new(Arc::new(data), metric, false));
        let criteria =
            vec![criteria::max_depth(3), criteria::min_cardinality(1)];
        let cluster =
            Cluster::new_root(Arc::clone(&dataset)).partition(&criteria);
        let (left, right) = cluster.children.read().unwrap().clone().unwrap();

        let left_ancestry = left.ancestry();
        assert_eq!(1, left_ancestry.len());

        let right_ancestry = right.ancestry();
        assert_eq!(1, right_ancestry.len());
    }
}
