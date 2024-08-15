//! A `Cluster` is a collection of "similar" instances in a dataset.

pub mod adapter;
mod ball;
mod lfd;
pub mod partition;

use std::hash::Hash;

use distances::Number;

use super::{dataset::ParDataset, Dataset, MetricSpace};

pub use ball::Ball;
pub use lfd::LFD;
pub use partition::Partition;

/// A `Cluster` is a collection of "similar" instances in a dataset.
///
/// # Type Parameters
///
/// - `I`: The type of the instances in the dataset.
/// - `U`: The type of the distance values between instances.
/// - `D`: The type of the dataset.
///
/// # Remarks
///
/// A `Cluster` must have certain properties to be useful in CLAM. These are:
///
/// - `depth`: The depth of the `Cluster` in the tree.
/// - `cardinality`: The number of instances in the `Cluster`.
/// - `indices`: The indices of the instances in the `Cluster`.
/// - `arg_center`: The index of the geometric median of the instances in the
///   `Cluster`. This may be computed exactly, using all instances in the
///   `Cluster`, or approximately, using a subset of the instances.
/// - `radius`: The distance from the center to the farthest instance in the
///   `Cluster`.
/// - `arg_radial`: The index of the instance that is farthest from the center.
/// - `lfd`: The Local Fractional Dimension of the `Cluster`.
///
/// A `Cluster` may have two or more children, which are `Cluster`s of the same
/// type. The children should be stored as a tuple with:
///
/// - The index of the extremal instance in the `Cluster` that was used to
///   create the child.
/// - The distance from that extremal instance to the farthest instance that was
///   assigned to the child. We refer to this as the "extent" of the child.
/// - The child `Cluster`.
pub trait Cluster<I, U: Number, D: Dataset<I, U>>: Ord + Hash + Sized {
    /// Deconstructs the `Cluster` into its most basic members. This is useful
    /// for adapting the `Cluster` to a different type of `Cluster`.
    ///
    /// # Returns
    ///
    /// - The `Cluster` itself, without the indices or children.
    /// - The indices of the instances in the `Cluster`.
    /// - The children of the `Cluster`.
    #[allow(clippy::type_complexity)]
    fn disassemble(self) -> (Self, Vec<usize>, Vec<(usize, U, Box<Self>)>);

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

    /// Gets the indices of the instances in the `Cluster`.
    fn indices(&self) -> impl Iterator<Item = usize> + '_;

    /// Sets the indices of the instances in the `Cluster`.
    fn set_indices(&mut self, indices: Vec<usize>);

    /// Returns the children of the `Cluster`.
    #[must_use]
    fn children(&self) -> &[(usize, U, Box<Self>)];

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)];

    /// Sets the children of the `Cluster`.
    fn set_children(&mut self, children: Vec<(usize, U, Self)>);

    /// Computes the distances from the `query` to all instances in the `Cluster`.
    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)>;

    /// Returns whether the `Cluster` is a descendant of another `Cluster`.
    fn is_descendant_of(&self, other: &Self) -> bool;

    /// Gets the child `Cluster`s.
    fn child_clusters<'a>(&'a self) -> impl Iterator<Item = &Self>
    where
        U: 'a,
    {
        self.children().iter().map(|(_, _, child)| child.as_ref())
    }

    /// Returns all `Cluster`s in the subtree of this `Cluster`, in depth-first order.
    fn subtree<'a>(&'a self) -> Vec<&'a Self>
    where
        U: 'a,
    {
        let mut clusters = vec![self];
        self.child_clusters().for_each(|child| clusters.extend(child.subtree()));
        clusters
    }

    /// Returns all leaf `Cluster`s in the subtree of this `Cluster`, in depth-first order.
    fn leaves<'a>(&'a self) -> Vec<&'a Self>
    where
        U: 'a,
    {
        self.subtree().into_iter().filter(|c| c.is_leaf()).collect()
    }

    /// Whether the `Cluster` is a leaf in the tree.
    fn is_leaf(&self) -> bool {
        self.children().is_empty()
    }

    /// Whether the `Cluster` is a singleton.
    fn is_singleton(&self) -> bool {
        self.cardinality() == 1 || self.radius() < U::EPSILON
    }

    /// Computes the distance from the `Cluster`'s center to a given `query`.
    fn distance_to_center(&self, data: &D, query: &I) -> U {
        let center = data.get(self.arg_center());
        MetricSpace::one_to_one(data, center, query)
    }

    /// Computes the distance from the `Cluster`'s center to another `Cluster`'s center.
    fn distance_to_other(&self, data: &D, other: &Self) -> U {
        Dataset::one_to_one(data, self.arg_center(), other.arg_center())
    }
}

/// A parallelized version of the `Cluster` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParCluster<I: Send + Sync, U: Number, D: ParDataset<I, U>>: Cluster<I, U, D> + Send + Sync {
    /// Parallelized version of the `distances_to_query` method.
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)>;
}
