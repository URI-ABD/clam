//! Traits for partitioning a `Cluster` and making trees.

use distances::Number;
use rayon::prelude::*;

use crate::{dataset::ParDataset, Dataset};

use super::{Cluster, ParCluster};

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively partitioned into a tree.
pub trait Partition<I, U: Number, D: Dataset<I, U>, C: Fn(&Self) -> bool>: Cluster<I, U, D> {
    /// Creates a new `Cluster` tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///   should be partitioned.
    /// - `seed`: An optional seed for random number generation.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the instances in the dataset.
    /// - `D`: The type of the dataset.
    /// - `C`: The type of the criteria function.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    fn new_tree(data: &D, criteria: &C, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::new(data, &indices, 0, seed);
        root.partition(data, indices, criteria, seed)
    }

    /// Partitions the `Cluster` into a number of child `Cluster`s.
    ///
    /// The number of children will be equal to the number of poles determined
    /// by the `find_poles` method.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `extrema`: The extrema to use for partitioning the `Cluster`.
    /// - `instances`: The instances in the `Cluster`.
    ///
    /// # Type Parameters
    ///
    /// - `I`: The type of the instances in the dataset.
    /// - `D`: The type of the dataset.
    ///
    /// # Returns
    ///
    /// - The instances with which to initialize the child `Cluster`s.
    /// - The distance from each pole to the farthest instance assigned to that
    ///   child.
    fn split_by_extrema(&self, data: &D, extrema: Vec<usize>, instances: Vec<usize>) -> (Vec<Vec<usize>>, Vec<U>) {
        // Find the distances from each pole to each instance.
        let polar_distances = Dataset::many_to_many(data, &extrema, &instances);

        // Convert the distances from row-major to column-major.
        let mut distances = vec![vec![U::ZERO; extrema.len()]; instances.len()];
        for (r, row) in polar_distances.iter().enumerate() {
            for (c, &(_, _, d)) in row.iter().enumerate() {
                distances[c][r] = d;
            }
        }

        // Initialize a child stack for each pole.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, U::ZERO)]).collect::<Vec<_>>();

        // For each pole, find the instances that are closer to it than to any
        // other pole.
        for (col, instance) in distances.iter().zip(instances) {
            let (pole_index, &d) = col
                .iter()
                .enumerate()
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[pole_index].push((instance, d));
        }

        child_stacks
            .into_iter()
            .map(|stack| {
                let (instances, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let max_distance = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (instances, max_distance)
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
    /// # Type Parameters
    ///
    /// - `I`: The type of the instances in the dataset.
    /// - `D`: The type of the dataset.
    /// - `C`: The type of the criteria function.
    ///
    /// # Returns
    ///
    /// - The root `Cluster` of the tree.
    /// - The instances in the `Cluster` in depth-first order of traversal of
    ///   the tree.
    #[must_use]
    fn partition(mut self, data: &D, mut indices: Vec<usize>, criteria: &C, seed: Option<u64>) -> Self {
        if !self.is_singleton() && criteria(&self) {
            let extrema = self.find_extrema(data);
            indices.retain(|i| !extrema.contains(i));
            let (child_stacks, child_extents) = self.split_by_extrema(data, extrema, indices);
            let depth = self.depth() + 1;
            let (children, other) = child_stacks
                .into_iter()
                .map(|child_indices| {
                    let (mut child, arg_r) = Self::new(data, &child_indices, depth, seed);
                    child = child.partition(data, child_indices, criteria, seed);
                    let child_indices = child.indices().collect::<Vec<_>>();
                    (child, (arg_r, child_indices))
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (arg_extrema, child_stacks) = other.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            indices = child_stacks.into_iter().flatten().collect::<Vec<_>>();
            let children = arg_extrema
                .into_iter()
                .zip(child_extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, c))
                .collect();
            self = self.set_children(children);
        };
        self.set_indices(indices);

        self
    }
}

/// `Cluster`s that use and provide parallelized methods.
#[allow(clippy::module_name_repetitions)]
pub trait ParPartition<I: Send + Sync, U: Number, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>:
    ParCluster<I, U, D>
{
    /// Parallelized version of the `new_tree` method.
    fn par_new_tree(data: &D, criteria: &C, seed: Option<u64>) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::par_new(data, &indices, 0, seed);
        root.par_partition(data, indices, criteria, seed)
    }

    /// Parallelized version of the `partition_once` method.
    fn par_split_by_extrema(&self, data: &D, extrema: Vec<usize>, instances: Vec<usize>) -> (Vec<Vec<usize>>, Vec<U>) {
        // Find the distances from each pole to each instance.
        let polar_distances = ParDataset::par_many_to_many(data, &extrema, &instances);

        // Convert the distances from row-major to column-major.
        let mut distances = vec![vec![U::ZERO; extrema.len()]; instances.len()];
        for (r, row) in polar_distances.iter().enumerate() {
            for (c, &(_, _, d)) in row.iter().enumerate() {
                distances[c][r] = d;
            }
        }

        // Initialize a child stack for each pole.
        let mut child_stacks = extrema.iter().map(|&p| vec![(p, U::ZERO)]).collect::<Vec<_>>();

        // For each pole, find the instances that are closer to it than to any
        // other pole.
        for (col, instance) in distances.iter().zip(instances) {
            let (pole_index, &d) = col
                .iter()
                .enumerate()
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[pole_index].push((instance, d));
        }

        let child_stacks: Vec<(Vec<usize>, U)> = child_stacks
            .into_par_iter()
            .map(|stack| {
                let (instances, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let max_distance = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (instances, max_distance)
            })
            .collect();

        child_stacks.into_iter().unzip::<_, _, Vec<_>, Vec<_>>()
    }

    /// Parallelized version of the `partition` method.
    #[must_use]
    fn par_partition(mut self, data: &D, mut indices: Vec<usize>, criteria: &C, seed: Option<u64>) -> Self {
        if !self.is_singleton() && criteria(&self) {
            let extrema = self.par_find_extrema(data);
            indices.retain(|i| !extrema.contains(i));
            let (child_stacks, child_extents) = self.par_split_by_extrema(data, extrema, indices);
            let depth = self.depth() + 1;
            let (children, other) = child_stacks
                .into_par_iter()
                .map(|child_indices| {
                    let (mut child, arg_r) = Self::par_new(data, &child_indices, depth, seed);
                    child = child.par_partition(data, child_indices, criteria, seed);
                    let child_indices = child.indices().collect::<Vec<_>>();
                    (child, (arg_r, child_indices))
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (arg_extrema, child_stacks) = other.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            indices = child_stacks.into_iter().flatten().collect::<Vec<_>>();
            let children = arg_extrema
                .into_iter()
                .zip(child_extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, c))
                .collect();
            self = self.set_children(children);
        };
        self.set_indices(indices);

        self
    }
}
