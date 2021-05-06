//! CLAM-Cluster
//!
//! Define and implement the `Cluster` struct.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;
use crate::utils::{argmax, argmin};
use criteria::PartitionCriterion;

const SUB_SAMPLE: usize = 100;

/// A 2-tuple of `Arc<Cluster>` representing the two child `Clusters`
/// formed when a `Cluster` is partitioned.
type Children<T, U> = (Arc<Cluster<T, U>>, Arc<Cluster<T, U>>);

// type ClusterMap<T, U> = HashMap<usize, Arc<Cluster<T, U>>>;

/// A collection of similar `Instances` from a `Dataset`.
///
/// `Clusters` can be unwieldy to use directly unless you have a
/// good grasp on the underlying invariants.
/// We anticipate that most users' needs will be well met by the higher-level
/// `Tree` and `Manifold` structs.
#[derive(Debug)]
pub struct Cluster<T: Number, U: Number> {
    /// An Arc to a struct that implements the Dataset trait.
    pub dataset: Arc<dyn Dataset<T, U>>,

    /// The name of a cluster and is meant to be unique in a tree.
    ///
    /// A Cluster's name is the turn when it would be visited in a breadth-first
    /// traversal of a perfect and balanced binary tree.
    /// The root is named 1 and all descendants follow.
    ///
    /// TODO: Switch to using a bit-vector to allow for trees with depth greater than 63.
    pub name: u64,

    /// The number of instances in this Cluster.
    pub cardinality: usize,

    /// The `Indices` (a Vec<usize>) of instances in this `Cluster`.
    pub indices: Indices,

    /// A subset of `indices' to use for fast, approximate calculations of center and radius.
    argsamples: Indices,

    /// The `Index` of the center of the Cluster.
    pub argcenter: Index,

    /// The `Index` of the instance in the Cluster that is farthest away from the center.
    pub argradius: Index,

    /// The distance from the center to the instance that is farthest away form the center.
    pub radius: U,

    /// Clusters start with no Children and may get Some after partition.
    pub children: Option<Children<T, U>>,
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
    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A reference to a struct that implements the `Dataset` trait.
    ///   We never copy `Instances` from the `Dataset`.
    /// * `name`: An Optional String name for this `Cluster`. Defaults to "1" for the root `Cluster`.
    /// * `indices`: The `Indices` of `Instances` from the dataset that are contained in the `Cluster`.
    pub fn new(dataset: Arc<dyn Dataset<T, U>>, name: u64, indices: Indices) -> Cluster<T, U> {
        let mut cluster = Cluster {
            dataset,
            name,
            cardinality: indices.len(),
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

    /// The depth of the `Cluster` in the tree. The root `Cluster` has depth 0.
    pub fn depth(&self) -> u8 {
        let size = std::mem::size_of::<usize>() << 3;
        let zeros = self.name.leading_zeros() as usize;
        (size - zeros - 1) as u8
    }

    /// A reference to the instance which is the (sometimes approximate) geometric median of the `Cluster`.
    ///
    /// TODO: Change the type of `Instance` to something generic.
    /// Ideally, something implementing an `Instance` trait so that `Dataset` becomes a collection of `Instances`.
    pub fn center(&self) -> ArrayView<T, IxDyn> {
        self.dataset.instance(self.argcenter)
    }

    /// Returns whether this `Cluster` contains only one unique `Instance` or its radius is 0.
    pub fn is_singleton(&self) -> bool {
        self.radius == U::from(0).unwrap()
    }

    /// Returns the distance from the center of the Cluster to the given instance.
    pub fn distance_to(&self, other: &Arc<Cluster<T, U>>) -> U {
        self.dataset.distance(self.argcenter, other.argcenter)
    }

    pub fn descend_towards(&self, _name: u64) -> Result<Arc<Cluster<T, U>>, String> {
        unimplemented!()
        // match self.children.borrow() {
        //     Some((left, right)) => {
        //         if left.name == cluster[0..left.depth()] {
        //             Ok(Arc::clone(left))
        //         } else if right.name == cluster[0..right.depth()] {
        //             Ok(Arc::clone(right))
        //         } else {
        //             Err(format!("Cluster {:} not found.", cluster))
        //         }
        //     }
        //     None => Err(format!("Cluster {:} not found.", cluster)),
        // }
    }

    /// Returns the indices of two maximally separated instances in the Cluster.
    fn poles(&self) -> (Index, Index) {
        let indices = self
            .indices
            .par_iter()
            .filter(|&&i| i != self.argradius)
            .cloned()
            .collect();
        let distances = self.dataset.distances_from(self.argradius, &indices);
        let (farthest, _) = argmax(&distances);
        (self.argradius, indices[farthest])
    }

    /// Recursively partition the cluster until some `criterion` determines
    /// that a leaf cluster has been reached.
    /// Returns a new cluster containing the built subtree.
    ///
    /// # Arguments
    ///
    /// * `partition_criteria`: A reference to `Vec<PartitionCriterion>`.
    ///   Each `Criterion` must evaluate to `true` otherwise the `Cluster`
    ///   cannot be partitioned.
    #[allow(clippy::ptr_arg)]
    pub fn partition(self, criteria: &Vec<PartitionCriterion<T, U>>) -> Cluster<T, U> {
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
        let (left, right): (Indices, Indices) = self
            .indices
            .par_iter()
            .partition(|&&i| self.dataset.distance(left, i) <= self.dataset.distance(i, right));

        // Ensure that left cluster is more populated than right cluster.
        let (left, right) = if right.len() > left.len() {
            (right, left)
        } else {
            (left, right)
        };

        // Recursively apply partition to child clusters.
        let (left, right) = rayon::join(
            || Cluster::new(Arc::clone(&self.dataset), self.name << 1, left).partition(criteria),
            || Cluster::new(Arc::clone(&self.dataset), 1 + (self.name << 1), right).partition(criteria),
        );

        // Return new cluster with the proper subtree
        Cluster {
            dataset: self.dataset,
            name: self.name,
            cardinality: self.cardinality,
            indices: self.indices,
            argsamples: self.argsamples,
            argcenter: self.argcenter,
            argradius: self.argradius,
            radius: self.radius,
            children: Some((Arc::new(left), Arc::new(right))),
        }
    }

    /// Returns the number of clusters in the subtree rooted at
    /// this cluster (excluding this cluster).
    pub fn num_descendants(&self) -> usize {
        match self.children.borrow() {
            Some((left, right)) => left.num_descendants() + right.num_descendants() + 2,
            None => 0,
        }
    }

    /// Returns unique samples from among Cluster.indices.
    ///
    /// These significantly speed up the computation of center and partition without much loss in accuracy.
    fn argsamples(&self) -> Indices {
        if self.cardinality <= SUB_SAMPLE {
            self.indices.clone()
        } else {
            let n = (self.cardinality as f64).sqrt() as Index;
            self.dataset.choose_unique(self.indices.clone(), n)
        }
    }

    /// Returns the `Index` of the `center` of the `Cluster`.
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

    /// Returns the `Index` of the `Instance` in the `Cluster` that is
    /// farthest away from the `center`.
    fn argradius(&self) -> Index {
        let distances = self.dataset.distances_from(self.argcenter, &self.indices);
        let (argradius, _) = argmax(&distances);
        self.indices[argradius]
    }

    /// Returns the distance from the `center` to the `Instance`
    /// in the `Cluster` that is farthest away from the `center.
    fn radius(&self) -> U {
        self.dataset.distance(self.argcenter, self.argradius)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{arr2, Array2};

    use crate::dataset::RowMajor;
    use crate::prelude::*;

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = arr2(&[[0., 0., 0.], [1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]);
        let dataset: Arc<dyn Dataset<f64, f64>> =
            Arc::new(RowMajor::<f64, f64>::new(data, "euclidean", false).unwrap());
        let criteria = vec![criteria::max_depth(3), criteria::min_cardinality(1)];
        let cluster = Cluster::new(Arc::clone(&dataset), 1, dataset.indices().clone()).partition(&criteria);

        assert_eq!(cluster, cluster);
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
            format!("argsamples: {:?},", cluster.argsamples),
            format!("argcenter: {:?},", cluster.argcenter),
            format!("argradius: {:?},", cluster.argradius),
            format!("radius: {:?},", cluster.radius),
            format!("children: {:?}", cluster.children),
            "}".to_string(),
        ]
        .join(" ");
        assert_eq!(format!("{:?}", cluster), cluster_str);

        let (left, right) = cluster.children.unwrap();
        assert_eq!(format!("{:}", left), "2");
        assert_eq!(format!("{:}", right), "3");

        for child in [left, right].iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality, 2);
            assert_eq!(child.num_descendants(), 2);
        }
    }
}
