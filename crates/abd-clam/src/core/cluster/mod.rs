//! A `Cluster` is a collection of "similar" items in a dataset.

use num::ToPrimitive;
use rayon::prelude::*;

mod ball;
mod lfd;
mod partition;

pub use ball::Ball;
pub use lfd::LFD;
pub use partition::{ParPartition, Partition};

use crate::{Dataset, DistanceValue, ParDataset};

/// A `Cluster` is a collection of "similar" items in a dataset.
///
/// It represents a metric ball in a metric space. All items in a `Cluster` are
/// within a certain distance of a center. The `Cluster` may have children,
/// which are `Cluster`s of the same type.
///
/// # Type Parameters
///
/// - `T`: The type of the distance values between items.
///
/// # Remarks
///
/// A `Cluster` must have certain properties to be useful in CLAM. These are:
///
/// - `depth`: The depth of the `Cluster` in the tree.
/// - `cardinality`: The number of items in the `Cluster`.
/// - `indices`: The indices into a dataset of the items in the `Cluster`.
/// - `arg_center`: The index of the geometric median of the items in the
///   `Cluster`. This may be computed exactly, using all items in the `Cluster`,
///   or approximately, using a subset of the items.
/// - `radius`: The distance from the center to the farthest item in the
///   `Cluster`.
/// - `arg_radial`: The index of the item that is farthest from the center.
/// - `lfd`: The Local Fractional Dimension of the `Cluster`.
///
/// A `Cluster` may have two or more children, which are `Cluster`s of the same
/// type. The children should be stored as a tuple with:
///
/// - The index of the extremal item in the `Cluster` that was used to
///   create the child.
/// - The distance from that extremal item to the farthest item that was
///   assigned to the child. We refer to this as the "extent" of the child.
/// - The child `Cluster`.
///
/// # Examples
///
/// See:
///
/// - [`Ball`](crate::core::cluster::Ball)
/// - [`PermutedBall`](crate::cakes::PermutedBall)
pub trait Cluster<T: DistanceValue>: Ord {
    /// Returns the depth of the `Cluster` in the tree.
    fn depth(&self) -> usize;

    /// Returns the cardinality of the `Cluster`.
    fn cardinality(&self) -> usize;

    /// Returns the index of the center item in the `Cluster`.
    fn arg_center(&self) -> usize;

    /// Sets the index of the center item in the `Cluster`.
    ///
    /// This is used to find the center item after permutation.
    fn set_arg_center(&mut self, arg_center: usize);

    /// Returns the radius of the `Cluster`.
    fn radius(&self) -> T;

    /// Returns the index of the radial item in the `Cluster`.
    fn arg_radial(&self) -> usize;

    /// Sets the index of the radial item in the `Cluster`.
    ///
    /// This is used to find the radial item after permutation.
    fn set_arg_radial(&mut self, arg_radial: usize);

    /// Returns the Local Fractional Dimension (LFD) of the `Cluster`.
    fn lfd(&self) -> f32;

    /// Returns whether this `Cluster` contains the given `index`ed point.
    fn contains(&self, idx: usize) -> bool;

    /// Gets the indices of the items in the `Cluster`.
    fn indices(&self) -> Vec<usize>;

    /// Sets the indices of the items in the `Cluster`.
    fn set_indices(&mut self, indices: &[usize]);

    /// Removes the indices of the items in this cluster.
    fn take_indices(&mut self) -> Vec<usize>;

    /// Returns the children of the `Cluster`.
    #[must_use]
    fn children(&self) -> Vec<&Self>;

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    fn children_mut(&mut self) -> Vec<&mut Self>;

    /// Sets the children of the `Cluster`.
    fn set_children(&mut self, children: Vec<Box<Self>>);

    /// Returns the owned children and sets the cluster's children to an empty vector.
    fn take_children(&mut self) -> Vec<Box<Self>>;

    /// Returns whether the `Cluster` is a descendant of another `Cluster`.
    fn is_descendant_of(&self, other: &Self) -> bool;

    /// Within a given tree, a `Cluster` must be uniquely identifiable by its
    /// `cardinality` and the index of any one of its items.
    fn unique_id(&self) -> (usize, usize) {
        (self.arg_center(), self.cardinality())
    }

    /// Trims the tree at the given depth. Returns the trimmed roots in the same
    /// order as the leaves of the trimmed tree at that depth.
    fn trim_at_depth(&mut self, depth: usize) -> Vec<Vec<Box<Self>>> {
        let mut queue = vec![self];
        let mut stack = Vec::new();

        while let Some(c) = queue.pop() {
            if c.depth() == depth {
                stack.push(c);
            } else {
                queue.extend(c.children_mut());
            }
        }

        stack.into_iter().map(Self::take_children).collect()
    }

    /// Inverts the `trim_at_depth` method.
    fn graft_at_depth(&mut self, depth: usize, trimmings: Vec<Vec<Box<Self>>>) {
        let mut queue = vec![self];
        let mut stack = Vec::new();

        while let Some(c) = queue.pop() {
            if c.depth() == depth {
                stack.push(c);
            } else {
                queue.extend(c.children_mut());
            }
        }

        stack
            .into_iter()
            .zip(trimmings)
            .for_each(|(c, children)| c.set_children(children));
    }

    /// Returns all `Cluster`s in the subtree of this `Cluster`, in depth-first order.
    fn subtree<'a>(&'a self) -> Vec<&'a Self>
    where
        T: 'a,
    {
        let mut clusters = vec![self];
        self.children()
            .into_iter()
            .for_each(|child| clusters.extend(child.subtree()));
        clusters
    }

    /// Returns all leaf `Cluster`s in the subtree of this `Cluster`, in depth-first order.
    fn leaves<'a>(&'a self) -> Vec<&'a Self>
    where
        T: 'a,
    {
        let mut queue = vec![self];
        let mut stack = vec![];

        while let Some(cluster) = queue.pop() {
            if cluster.is_leaf() {
                stack.push(cluster);
            } else {
                queue.extend(cluster.children());
            }
        }

        stack
    }

    /// Returns mutable references to all leaf `Cluster`s in the subtree of this `Cluster`, in depth-first order.
    fn leaves_mut<'a>(&'a mut self) -> Vec<&'a mut Self>
    where
        T: 'a,
    {
        let mut queue = vec![self];
        let mut stack = vec![];

        while let Some(cluster) = queue.pop() {
            if cluster.is_leaf() {
                stack.push(cluster);
            } else {
                queue.extend(cluster.children_mut());
            }
        }

        stack
    }

    /// Whether the `Cluster` is a leaf in the tree.
    fn is_leaf(&self) -> bool {
        self.children().is_empty()
    }

    /// Whether the `Cluster` is a singleton.
    fn is_singleton(&self) -> bool {
        self.cardinality() == 1 || self.radius() == T::zero()
    }

    /// Returns the expected radius for `k` items from the cluster center, using
    /// the cluster's LFD to make the estimate.
    fn radius_for_k(&self, k: usize) -> f32 {
        let r = self
            .radius()
            .to_f32()
            .unwrap_or_else(|| unreachable!("Radius is not finite"));
        if self.cardinality() == k {
            r
        } else {
            let car = self
                .cardinality()
                .to_f32()
                .unwrap_or_else(|| unreachable!("Cardinality is not finite"));
            let k = k.to_f32().unwrap_or_else(|| unreachable!("k is not finite"));
            r * (k / car).powf(self.lfd().recip())
        }
    }

    /// Returns the distance between the centers of two `Cluster`s.
    fn distance_to<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(&self, other: &Self, data: &D, metric: &M) -> T {
        data.one_to_one(self.arg_center(), other.arg_center(), metric)
    }

    /// Returns whether this cluster has any overlap with a query ball, along
    /// with the distance from the query to the cluster center.
    fn overlaps_with<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &self,
        data: &D,
        metric: &M,
        query: &I,
        radius: T,
    ) -> (bool, T) {
        let d = data.query_to_one(query, self.arg_center(), metric);
        (d <= self.radius() + radius, d)
    }

    /// Returns only those children of the `Cluster` that overlap with a query
    /// ball, along with the distance from the query to each child's center.
    fn overlapping_children<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &self,
        data: &D,
        metric: &M,
        query: &I,
        radius: T,
    ) -> Vec<(&Self, T)> {
        self.children()
            .into_iter()
            .map(|c| (c, c.overlaps_with(data, metric, query, radius)))
            .filter(|&(_, (o, _))| o)
            .map(|(c, (_, d))| (c, d))
            .collect()
    }
}

/// A parallelized version of the `Cluster` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParCluster<T: DistanceValue + Send + Sync>: Cluster<T> + Send + Sync {
    /// Parallel version of [`Cluster::indices`](crate::core::cluster::Cluster::indices).
    fn par_indices(&self) -> impl ParallelIterator<Item = usize>;

    /// Parallel version of [`Cluster::overlapping_children`](crate::core::cluster::Cluster::overlapping_children).
    fn par_overlapping_children<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
        query: &I,
        radius: T,
    ) -> Vec<(&Self, T)> {
        self.children()
            .into_par_iter()
            .map(|c| (c, c.overlaps_with(data, metric, query, radius)))
            .filter(|&(_, (o, _))| o)
            .map(|(c, (_, d))| (c, d))
            .collect()
    }
}
