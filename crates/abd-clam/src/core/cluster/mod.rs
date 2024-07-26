//! A `Cluster` is a collection of "similar" instances in a dataset.

mod adaptors;
mod ball;
mod children;
mod index_store;
mod lfd;
mod partition;

use core::fmt::Debug;

use distances::Number;

use super::{Dataset, MetricSpace, ParDataset};

#[allow(clippy::module_name_repetitions)]
pub use adaptors::ClusterAdaptor;
pub use ball::Ball;
pub use children::Children;
pub use index_store::IndexStore;
pub use lfd::LFD;
pub use partition::{ParPartition, Partition};

/// A `Cluster` is a collection of "similar" instances in a dataset.
///
/// # Type Parameters
///
/// - `U`: The type of the distance values between instances.
/// - `P`: The type of the parameters used to create the `Cluster`.
pub trait Cluster<U: Number>: Debug + PartialOrd {
    /// Creates a new `Cluster`.
    ///
    /// This should store indices as `IndexStore::EveryCluster`.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `indices`: The indices of instances in the `Cluster`.
    /// - `depth`: The depth of the `Cluster` in the tree.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the instances in the dataset.
    /// - `D`: The type of the dataset.
    ///
    /// # Returns
    ///
    /// - The new `Cluster`.
    /// - The index of the radial instance in `instances`.
    fn new<I, D: Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
    where
        Self: Sized;

    /// Returns the depth os the `Cluster` in the tree.
    fn depth(&self) -> usize;

    /// Returns the cardinality of the `Cluster`.
    fn cardinality(&self) -> usize;

    /// Returns the index of the center instance in the `Cluster`.
    fn arg_center(&self) -> usize;

    /// Sets the index of the center instance in the `Cluster`.
    ///
    /// This is used to find the center instance after permutation.
    fn set_arg_center(&mut self, arg_center: usize);

    /// Returns the radius of the `Cluster`.
    fn radius(&self) -> U;

    /// Returns the index of the radial instance in the `Cluster`.
    fn arg_radial(&self) -> usize;

    /// Sets the index of the radial instance in the `Cluster`.
    ///
    /// This is used to find the radial instance after permutation.
    fn set_arg_radial(&mut self, arg_radial: usize);

    /// Returns the Local Fractional Dimension (LFD) of the `Cluster`.
    fn lfd(&self) -> f32;

    /// Returns the `IndexStore` of the `Cluster`.
    fn index_store(&self) -> &IndexStore;

    /// Sets the `IndexStore` of the `Cluster`.
    fn set_index_store(&mut self, indices: IndexStore);

    /// Returns the children of the `Cluster`.
    #[must_use]
    fn children(&self) -> Option<&Children<U, Self>>
    where
        Self: Sized;

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    fn children_mut(&mut self) -> Option<&mut Children<U, Self>>
    where
        Self: Sized;

    /// Sets the children of the `Cluster`.
    #[must_use]
    fn set_children(self, children: Children<U, Self>) -> Self
    where
        Self: Sized;

    /// Finds the extrema of the `Cluster`.
    ///
    /// The extrema are meant to be well-separated instances that can be used to
    /// partition the `Cluster` into some number of child `Cluster`s. The number
    /// of children will be equal to the number of extrema determined by this
    /// method.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the instances in the dataset.
    /// - `D`: The type of the dataset.
    ///
    /// # Returns
    ///
    /// - The extrema to use for partitioning the `Cluster`.
    /// - The remaining instances in the `Cluster`.
    /// - The pairwise distances between the extrema.
    fn find_extrema<I, D: Dataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>);

    /// Whether the `Cluster` is a leaf in the tree.
    fn is_leaf(&self) -> bool
    where
        Self: Sized,
    {
        self.children().is_none()
    }

    /// Whether the `Cluster` is a singleton.
    fn is_singleton(&self) -> bool {
        self.cardinality() == 1 || self.radius() < U::EPSILON
    }

    /// Gets the indices of the instances in the `Cluster`.
    fn indices(&self) -> Vec<usize>
    where
        Self: Sized,
    {
        self.index_store().indices(self)
    }

    /// Returns the given distance repeated with the indices of the instances in
    /// the `Cluster`.
    fn repeat_distance(&self, d: U) -> Vec<(usize, U)>
    where
        Self: Sized,
    {
        self.index_store().repeat_distance(self, d)
    }

    /// Computes the distances from the `query` to all instances in the `Cluster`.
    fn distances<I, D: Dataset<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)>
    where
        Self: Sized,
    {
        self.index_store().distances(data, query, self)
    }

    /// Parallelized version of the `distances` method.
    fn par_distances<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)>
    where
        Self: Sized + Send + Sync,
    {
        self.index_store().par_distances(data, query, self)
    }

    /// Computes the distance from the `Cluster`'s center to a given `query`.
    fn distance_to_center<I, D: Dataset<I, U>>(&self, data: &D, query: &I) -> U {
        let center = data.get(self.arg_center());
        MetricSpace::one_to_one(data, center, query)
    }

    /// Computes the distance from the `Cluster`'s center to another `Cluster`'s center.
    fn distance_to_other<I, D: Dataset<I, U>>(&self, data: &D, other: &Self) -> U {
        Dataset::one_to_one(data, self.arg_center(), other.arg_center())
    }
}

/// A parallelized version of the `Cluster` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParCluster<U: Number>: Cluster<U> + Send + Sync {
    /// Parallelized version of the `new` method.
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize)
    where
        Self: Sized;

    /// Parallelized version of the `find_extrema` method.
    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>);
}
