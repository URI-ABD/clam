//! Traits for partitioning a `Cluster` and making trees.

use distances::Number;
use rayon::prelude::*;

use crate::{dataset::ParDataset, Dataset};

use super::{Cluster, ParCluster};

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively
/// partitioned into a tree.
///
/// # Type Parameters
///
/// - `I`: The type of the instances in the `Dataset`.
/// - `U`: The type of the distance values.
/// - `D`: The type of the `Dataset`.
pub trait Partition<I, U: Number, D: Dataset<I, U>>: Cluster<I, U, D> {
    /// Creates a new `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `indices`: The indices of instances in the `Cluster`.
    /// - `depth`: The depth of the `Cluster` in the tree.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Returns
    ///
    /// - The new `Cluster`.
    fn new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self;

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
    /// # Returns
    ///
    /// The extrema to use for partitioning the `Cluster`.
    fn find_extrema(&self, data: &D) -> Vec<usize>;

    /// Creates a new `Cluster` tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///    should be partitioned. A `Cluster` will only be partitioned if it is
    ///    not a singleton and this function returns `true`.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    fn new_tree<C: Fn(&Self) -> bool>(data: &D, criteria: &C, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let mut root = Self::new(data, &indices, 0, seed);
        root.partition(data, criteria, seed);
        root
    }

    /// Partitions the `Cluster` into a number of child `Cluster`s.
    ///
    /// The number of children will be equal to the number of extrema determined
    /// by the `find_extrema` method.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `extrema`: The indices of extrema for partitioning the `Cluster`.
    /// - `instances`: The indices of instances in the `Cluster`.
    ///
    /// # Returns
    ///
    /// - The indices of instances with which to initialize the children. The
    ///   0th element of each inner `Vec` is the index of the corresponding
    ///   extremum.
    /// - The distance from each extremum to the farthest instance assigned to
    ///   that child, i.e. the "extent" of the child.
    fn split_by_extrema(&self, data: &D, extrema: &[usize]) -> (Vec<Vec<usize>>, Vec<U>) {
        let instances = self.indices().filter(|i| !extrema.contains(i)).collect::<Vec<_>>();

        // Find the distances from each extremum to each instance.
        let extremal_distances = Dataset::many_to_many(data, extrema, &instances);

        // Convert the distances from row-major to column-major.
        let distances = {
            let mut distances = vec![vec![U::ZERO; extrema.len()]; instances.len()];
            for (r, row) in extremal_distances.into_iter().enumerate() {
                for (c, (_, _, d)) in row.into_iter().enumerate() {
                    distances[c][r] = d;
                }
            }
            distances
        };

        // Initialize a child stack for each extremum.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, U::ZERO)]).collect::<Vec<_>>();

        // For each extremum, find the instances that are closer to it than to
        // any other extremum.
        for (col, instance) in distances.into_iter().zip(instances) {
            let (e_index, d) = col
                .into_iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[e_index].push((instance, d));
        }

        child_stacks
            .into_iter()
            .map(|stack| {
                let (indices, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let extent = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (indices, extent)
            })
            .unzip()
    }

    /// Recursively partitions the `Cluster` into a tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///   should be partitioned.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    /// - The instances in the `Cluster` in depth-first order of traversal of
    ///   the tree.
    fn partition<C: Fn(&Self) -> bool>(&mut self, data: &D, criteria: &C, seed: Option<u64>) {
        if !self.is_singleton() && criteria(self) {
            ftlog::debug!(
                "Starting `partition` of a cluster at depth {}, with {} instances.",
                self.depth(),
                self.cardinality()
            );

            // Find the extrema.
            let extrema = self.find_extrema(data);
            // Split the instances by the extrema.
            let (child_stacks, child_extents) = self.split_by_extrema(data, &extrema);
            // Increment the depth for the children.
            let depth = self.depth() + 1;
            // Create the children.
            let (children, other) = child_stacks
                .into_iter()
                .map(|child_indices| {
                    let mut child = Self::new(data, &child_indices, depth, seed);
                    let arg_r = child.arg_radial();
                    child.partition(data, criteria, seed);
                    let child_indices = child.indices().collect::<Vec<_>>();
                    (Box::new(child), (arg_r, child_indices))
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (arg_extrema, child_stacks) = other.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

            // Recombine the indices.
            self.set_indices(child_stacks.into_iter().flatten().collect());

            // Combine the children with the extrema and extents.
            let children = arg_extrema
                .into_iter()
                .zip(child_extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, c))
                .collect();
            // Update the `Cluster`'s children.
            self.set_children(children);

            ftlog::debug!(
                "Finished `partition` of a cluster at depth {}, with {} instances.",
                self.depth(),
                self.cardinality()
            );
        };
    }

    /// Partitions the leaf `Cluster`s the tree even further using a different
    /// criteria.
    fn partition_further<C: Fn(&Self) -> bool>(&mut self, data: &D, criteria: &C, seed: Option<u64>) {
        self.leaves_mut()
            .into_iter()
            .for_each(|child| child.partition(data, criteria, seed));
    }
}

/// `Cluster`s that use and provide parallelized methods.
#[allow(clippy::module_name_repetitions)]
pub trait ParPartition<I: Send + Sync, U: Number, D: ParDataset<I, U>>:
    ParCluster<I, U, D> + Partition<I, U, D>
{
    /// Parallelized version of the `new` method.
    fn par_new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self;

    /// Parallelized version of the `find_extrema` method.
    fn par_find_extrema(&self, data: &D) -> Vec<usize>;

    /// Parallelized version of the `new_tree` method.
    fn par_new_tree<C: (Fn(&Self) -> bool) + Send + Sync>(data: &D, criteria: &C, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let mut root = Self::par_new(data, &indices, 0, seed);
        root.par_partition(data, criteria, seed);
        root
    }

    /// Parallelized version of the `partition_once` method.
    fn par_split_by_extrema(&self, data: &D, extrema: &[usize]) -> (Vec<Vec<usize>>, Vec<U>) {
        let instances = self.indices().filter(|i| !extrema.contains(i)).collect::<Vec<_>>();
        // Find the distances from each extremum to each instance.
        let extremal_distances = ParDataset::par_many_to_many(data, extrema, &instances);

        // Convert the distances from row-major to column-major.
        let distances = {
            let mut distances = vec![vec![U::ZERO; extrema.len()]; instances.len()];
            for (r, row) in extremal_distances.into_iter().enumerate() {
                for (c, (_, _, d)) in row.into_iter().enumerate() {
                    distances[c][r] = d;
                }
            }
            distances
        };

        // Initialize a child stack for each extremum.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, U::ZERO)]).collect::<Vec<_>>();

        // For each extremum, find the instances that are closer to it than to
        // any other extremum.
        for (col, instance) in distances.into_iter().zip(instances) {
            let (e_index, d) = col
                .into_iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[e_index].push((instance, d));
        }

        // Find the maximum distance for each child and return the instances.
        child_stacks
            .into_par_iter()
            .map(|stack| {
                let (indices, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let extent = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (indices, extent)
            })
            .unzip()
    }

    /// Parallelized version of the `partition` method.
    fn par_partition<C: (Fn(&Self) -> bool) + Send + Sync>(&mut self, data: &D, criteria: &C, seed: Option<u64>) {
        if !self.is_singleton() && criteria(self) {
            ftlog::debug!(
                "Starting `par_partition` of a cluster at depth {}, with {} instances.",
                self.depth(),
                self.cardinality()
            );

            // Find the extrema.
            let extrema = self.par_find_extrema(data);
            // Split the instances by the extrema.
            let (child_stacks, child_extents) = self.par_split_by_extrema(data, &extrema);
            // Increment the depth for the children.
            let depth = self.depth() + 1;

            // Create the children.
            let (children, other) = child_stacks
                .into_par_iter()
                .map(|child_indices| {
                    let e_index = child_indices[0];
                    let mut child = Self::par_new(data, &child_indices, depth, seed);
                    child.par_partition(data, criteria, seed);
                    let child_indices = child.indices().collect::<Vec<_>>();
                    (Box::new(child), (e_index, child_indices))
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (arg_extrema, child_stacks) = other.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            // Recombine the indices from the children.
            self.set_indices(child_stacks.into_iter().flatten().collect());

            // Combine the children with the extrema and extents.
            let children = arg_extrema
                .into_iter()
                .zip(child_extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, c))
                .collect();
            // Update the `Cluster`'s children.
            self.set_children(children);

            ftlog::debug!(
                "Finished `par_partition` of a cluster at depth {}, with {} instances.",
                self.depth(),
                self.cardinality()
            );
        };
    }

    /// Parallelized version of the `partition_further` method.
    fn par_partition_further<C: (Fn(&Self) -> bool) + Send + Sync>(
        &mut self,
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) {
        self.leaves_mut()
            .into_par_iter()
            .for_each(|child| child.par_partition(data, criteria, seed));
    }
}
