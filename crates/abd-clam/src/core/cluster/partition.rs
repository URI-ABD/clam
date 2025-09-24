//! Traits for partitioning a `Cluster` and making trees.

use rayon::prelude::*;

use crate::{utils, Dataset, ParDataset};

use super::{Cluster, DistanceValue, ParCluster};

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively
/// partitioned into a tree.
///
/// # Type Parameters
///
/// - `T`: The type of the distance values.
///
/// # Examples
///
/// See:
///
/// - [`Ball`](crate::core::cluster::Ball)
/// - [`BalancedBall`](crate::core::cluster::BalancedBall)
pub trait Partition<T: DistanceValue>: Cluster<T> + Sized {
    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculations.
    /// - `indices`: The indices of items in the `Cluster`.
    /// - `depth`: The depth of the `Cluster` in the tree.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The items in the dataset.
    /// - `D`: The dataset.
    /// - `M`: The metric.
    ///
    /// # Errors
    ///
    /// - If the `indices` are empty.
    /// - Any error that occurs when creating the `Cluster` depending on the
    ///   implementation.
    fn new<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
    ) -> Result<Self, String>;

    /// Finds the extrema of the `Cluster`.
    ///
    /// The extrema are meant to be well-separated items that can be used to
    /// partition the `Cluster` into some number of child `Cluster`s. The number
    /// of children will be equal to the number of extrema determined by this
    /// method.
    ///
    /// There will be panics if this method returns less than two extrema.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculations.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The items in the dataset.
    /// - `D`: The dataset.
    /// - `M`: The metric.
    fn find_extrema<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(&self, data: &D, metric: &M) -> Vec<usize>;

    /// Creates a new `Cluster` tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculations.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///   should be partitioned. A `Cluster` will only be partitioned if it is
    ///   not a singleton and this function returns `true`.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The items in the dataset.
    /// - `D`: The dataset.
    /// - `M`: The metric.
    /// - `C`: The criteria function for partitioning.
    fn new_tree<I, D: Dataset<I>, M: Fn(&I, &I) -> T, C: Fn(&Self) -> bool>(data: &D, metric: &M, criteria: &C) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let mut root = Self::new(data, metric, &indices, 0)
            .unwrap_or_else(|e| unreachable!("We ensured that the indices are not empty: {e}"));
        root.partition(data, metric, criteria);
        root
    }

    /// Creates a new `Cluster` tree using an iterative partitioning method to
    /// avoid stack overflows from the lack of tail call optimization.
    fn new_tree_iterative<I, D: Dataset<I>, M: Fn(&I, &I) -> T, C: Fn(&Self) -> bool>(
        data: &D,
        metric: &M,
        criteria: &C,
        depth_stride: usize,
    ) -> Self {
        let mut target_depth = depth_stride;
        let stride_criteria = |c: &Self| c.depth() < target_depth && criteria(c);

        let mut root = Self::new_tree(data, metric, &stride_criteria);

        let mut stride_leaves = root
            .leaves_mut()
            .into_iter()
            .filter(|c| (c.depth() == depth_stride) && criteria(c))
            .collect::<Vec<_>>();
        while !stride_leaves.is_empty() {
            target_depth += depth_stride;
            let stride_criteria = |c: &Self| c.depth() < target_depth && criteria(c);
            for c in stride_leaves {
                c.partition(data, metric, &stride_criteria);
            }
            stride_leaves = root
                .leaves_mut()
                .into_iter()
                .filter(|c| (c.depth() == target_depth) && criteria(c))
                .collect::<Vec<_>>();
        }

        root
    }

    /// Partitions the `Cluster` into a number of child `Cluster`s.
    ///
    /// The number of children will be equal to the number of extrema determined
    /// by the `find_extrema` method.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculations.
    /// - `extrema`: The indices of extrema for partitioning the `Cluster`.
    /// - `items`: The indices of items in the `Cluster`.
    ///
    /// # Returns
    ///
    /// - The indices of items with which to initialize the children. The
    ///   0th element of each inner `Vec` is the index of the corresponding
    ///   extremum.
    /// - The distance from each extremum to the farthest item assigned to
    ///   that child, i.e. the "extent" of the child.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The items in the dataset.
    /// - `D`: The dataset.
    /// - `M`: The metric.
    fn split_by_extrema<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(
        &self,
        data: &D,
        metric: &M,
        extrema: &[usize],
    ) -> Vec<Vec<usize>> {
        let items = self
            .indices()
            .into_iter()
            .filter(|i| !extrema.contains(i))
            .collect::<Vec<_>>();

        // Find the distances from each extremum to each item.
        let extremal_distances = data.many_to_many(extrema, &items, metric);

        // Convert the distances from row-major to column-major.
        let distances = {
            let mut distances = vec![vec![T::zero(); extrema.len()]; items.len()];
            for (r, row) in extremal_distances.into_iter().enumerate() {
                for (c, (_, _, d)) in row.into_iter().enumerate() {
                    distances[c][r] = d;
                }
            }
            distances
        };

        // Initialize a child stack for each extremum.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, T::zero())]).collect::<Vec<_>>();

        // For each extremum, find the items that are closer to it than to
        // any other extremum.
        for (col, item) in distances.into_iter().zip(items) {
            let (e_index, d) = utils::arg_min(&col).unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[e_index].push((item, d));
        }

        child_stacks
            .into_iter()
            .map(|stack| stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>().0)
            .collect()
    }

    /// Partitions the `Cluster` once instead of recursively.
    fn partition_once<I, D: Dataset<I>, M: Fn(&I, &I) -> T>(&self, data: &D, metric: &M) -> Vec<Box<Self>> {
        // Find the extrema.
        let extrema = self.find_extrema(data, metric);
        // Split the items by the extrema.
        let child_stacks = self.split_by_extrema(data, metric, &extrema);
        // Increment the depth for the children.
        let depth = self.depth() + 1;

        // Create the children.
        child_stacks
            .into_iter()
            .map(|child_indices| {
                Self::new(data, metric, &child_indices, depth)
                    .unwrap_or_else(|e| unreachable!("We ensured that the indices are not empty: {e}"))
            })
            .map(Box::new)
            .collect()
    }

    /// Recursively partitions the `Cluster` into a tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the items.
    /// - `metric`: The metric to use for distance calculations.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///   should be partitioned.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    /// - The items in the `Cluster` in depth-first order of traversal of
    ///   the tree.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The items in the dataset.
    /// - `D`: The dataset.
    /// - `M`: The metric.
    /// - `C`: The criteria function for partitioning.
    fn partition<I, D: Dataset<I>, M: Fn(&I, &I) -> T, C: Fn(&Self) -> bool>(
        &mut self,
        data: &D,
        metric: &M,
        criteria: &C,
    ) {
        if !self.is_singleton() && criteria(self) {
            ftlog::trace!(
                "Starting `partition` of a cluster at depth {}, with {} items.",
                self.depth(),
                self.cardinality()
            );

            let mut children = self.partition_once(data, metric);
            for child in &mut children {
                child.partition(data, metric, criteria);
            }
            let indices = children.iter().flat_map(|c| c.indices()).collect::<Vec<_>>();
            self.set_indices(&indices);
            self.set_children(children);

            ftlog::trace!(
                "Finished `partition` of a cluster at depth {}, with {} items.",
                self.depth(),
                self.cardinality()
            );
        }
    }
}

/// `Cluster`s that use and provide parallelized methods.
///
/// # Examples
///
/// See:
///
/// - [`Ball`](crate::core::cluster::Ball)
/// - [`BalancedBall`](crate::core::cluster::BalancedBall)
#[allow(clippy::module_name_repetitions)]
pub trait ParPartition<T: DistanceValue + Send + Sync>: ParCluster<T> + Partition<T> {
    /// Parallelized version of [`Partition::new`](crate::core::cluster::Partition::new).
    ///
    /// # Errors
    ///
    /// See [`Partition::new`](crate::core::cluster::Partition::new).
    fn par_new<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        data: &D,
        metric: &M,
        indices: &[usize],
        depth: usize,
    ) -> Result<Self, String>;

    /// Parallelized version of [`Partition::find_extrema`](crate::core::cluster::Partition::find_extrema).
    fn par_find_extrema<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
    ) -> Vec<usize>;

    /// Parallelized version of [`Partition::new_tree`](crate::core::cluster::Partition::new_tree).
    fn par_new_tree<
        I: Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        C: (Fn(&Self) -> bool) + Send + Sync,
    >(
        data: &D,
        metric: &M,
        criteria: &C,
    ) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let mut root = Self::par_new(data, metric, &indices, 0)
            .unwrap_or_else(|e| unreachable!("We ensured that the indices are not empty: {e}"));
        root.par_partition(data, metric, criteria);
        root
    }

    /// Parallelized version of [`Partition::new_tree_iterative`](crate::core::cluster::Partition::new_tree_iterative).
    fn par_new_tree_iterative<
        I: Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        C: (Fn(&Self) -> bool) + Send + Sync,
    >(
        data: &D,
        metric: &M,
        criteria: &C,
        depth_stride: usize,
    ) -> Self {
        let mut target_depth = depth_stride;
        let stride_criteria = |c: &Self| c.depth() < target_depth && criteria(c);

        let mut root = Self::par_new_tree(data, metric, &stride_criteria);

        let mut stride_leaves = root
            .leaves_mut()
            .into_par_iter()
            .filter(|c| (c.depth() == depth_stride) && criteria(c))
            .collect::<Vec<_>>();
        while !stride_leaves.is_empty() {
            target_depth += depth_stride;
            let stride_criteria = |c: &Self| c.depth() < target_depth && criteria(c);
            stride_leaves
                .into_par_iter()
                .for_each(|c| c.par_partition(data, metric, &stride_criteria));
            stride_leaves = root
                .leaves_mut()
                .into_par_iter()
                .filter(|c| (c.depth() == target_depth) && criteria(c))
                .collect::<Vec<_>>();
        }

        root
    }

    /// Parallelized version of [`Partition::split_by_extrema`](crate::core::cluster::Partition::split_by_extrema).
    fn par_split_by_extrema<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
        extrema: &[usize],
    ) -> Vec<Vec<usize>> {
        let items = self
            .indices()
            .into_iter()
            .filter(|i| !extrema.contains(i))
            .collect::<Vec<_>>();
        // Find the distances from each extremum to each item.
        let extremal_distances = data.par_many_to_many(extrema, &items, metric);

        // Convert the distances from row-major to column-major.
        let distances = {
            let mut distances = vec![vec![T::zero(); extrema.len()]; items.len()];
            for (r, row) in extremal_distances.into_iter().enumerate() {
                for (c, (_, _, d)) in row.into_iter().enumerate() {
                    distances[c][r] = d;
                }
            }
            distances
        };

        // Initialize a child stack for each extremum.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, T::zero())]).collect::<Vec<_>>();

        // For each extremum, find the items that are closer to it than to
        // any other extremum.
        for (col, item) in distances.into_iter().zip(items) {
            let (e_index, d) = utils::arg_min(&col).unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[e_index].push((item, d));
        }

        child_stacks
            .into_iter()
            .map(|stack| stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>().0)
            .collect()
    }

    /// Parallelized version of [`Partition::partition_once`](crate::core::cluster::Partition::partition_once).
    fn par_partition_once<I: Send + Sync, D: ParDataset<I>, M: (Fn(&I, &I) -> T) + Send + Sync>(
        &self,
        data: &D,
        metric: &M,
    ) -> Vec<Box<Self>> {
        // Find the extrema.
        let extrema = self.par_find_extrema(data, metric);
        // Split the items by the extrema.
        let child_stacks = self.par_split_by_extrema(data, metric, &extrema);
        // Increment the depth for the children.
        let depth = self.depth() + 1;

        // Create the children.
        child_stacks
            .into_par_iter()
            .map(|child_indices| {
                Self::par_new(data, metric, &child_indices, depth)
                    .unwrap_or_else(|e| unreachable!("We ensured that the indices are not empty: {e}"))
            })
            .map(Box::new)
            .collect()
    }

    /// Parallelized version of [`Partition::partition`](crate::core::cluster::Partition::partition).
    fn par_partition<
        I: Send + Sync,
        D: ParDataset<I>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        C: (Fn(&Self) -> bool) + Send + Sync,
    >(
        &mut self,
        data: &D,
        metric: &M,
        criteria: &C,
    ) {
        if !self.is_singleton() && criteria(self) {
            ftlog::trace!(
                "Starting `par_partition` of a cluster at depth {}, with {} items.",
                self.depth(),
                self.cardinality()
            );

            let mut children = self.par_partition_once(data, metric);
            children.par_iter_mut().for_each(|child| {
                child.par_partition(data, metric, criteria);
            });
            let indices = children.iter().flat_map(|c| c.indices()).collect::<Vec<_>>();
            self.set_indices(&indices);
            self.set_children(children);

            ftlog::trace!(
                "Finished `par_partition` of a cluster at depth {}, with {} items.",
                self.depth(),
                self.cardinality()
            );
        }
    }
}
