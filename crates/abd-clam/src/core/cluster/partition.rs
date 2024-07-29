//! Traits for partitioning a `Cluster` and making trees.

use distances::Number;
use rayon::prelude::*;

use crate::{dataset::ParDataset, Dataset};

use super::{Children, Cluster, ParCluster};

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively partitioned into a tree.
pub trait Partition<U: Number>: Cluster<U> {
    /// Creates a new `Cluster` tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `criteria`: The function to use for determining when a `Cluster`
    /// should be partitioned.
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
    fn new_tree<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(data: &D, criteria: &C, seed: Option<u64>) -> Self {
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
    /// The instances with which to initialize the child `Cluster`s.
    fn split_by_extrema<I, D: Dataset<I, U>>(
        &self,
        data: &D,
        extrema: Vec<usize>,
        instances: Vec<usize>,
    ) -> Vec<Vec<usize>> {
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
        let mut child_stacks = extrema.iter().map(|&p| vec![p]).collect::<Vec<_>>();

        // For each pole, find the instances that are closer to it than to any
        // other pole.
        for (col, instance) in distances.iter().zip(instances) {
            let (pole_index, _) = col
                .iter()
                .enumerate()
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[pole_index].push(instance);
        }

        child_stacks
    }

    /// Recursively partitions the `Cluster` into a tree.
    ///
    /// # Arguments
    ///
    /// - `data`: The dataset containing the instances.
    /// - `criteria`: The function to use for determining when a `Cluster`
    ///  should be partitioned.
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
    ///  the tree.
    #[must_use]
    fn partition<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(
        mut self,
        data: &D,
        mut indices: Vec<usize>,
        criteria: &C,
        seed: Option<u64>,
    ) -> Self {
        indices = if !self.is_singleton() && criteria(&self) {
            let (extrema, mut indices, extremal_distances) = self.find_extrema(data);
            let child_stacks = self.split_by_extrema(data, extrema, indices);
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
            let children = Children::new(children, arg_extrema, extremal_distances);
            self = self.set_children(children);
            indices
        } else {
            indices
        };
        self.set_indices(indices);

        self
    }
}

/// `Cluster`s that use and provide parallelized methods.
#[allow(clippy::module_name_repetitions)]
pub trait ParPartition<U: Number>: ParCluster<U> {
    /// Parallelized version of the `new_tree` method.
    fn par_new_tree<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> Self {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::par_new(data, &indices, 0, seed);
        root.par_partition(data, indices, criteria, seed)
    }

    /// Parallelized version of the `partition_once` method.
    fn par_split_by_extrema<I: Send + Sync, D: ParDataset<I, U>>(
        &self,
        data: &D,
        extrema: Vec<usize>,
        instances: Vec<usize>,
    ) -> Vec<Vec<usize>> {
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
        let mut child_stacks = extrema.iter().map(|&p| vec![p]).collect::<Vec<_>>();

        // For each pole, find the instances that are closer to it than to any
        // other pole.
        for (col, instance) in distances.iter().zip(instances) {
            let (pole_index, _) = col
                .iter()
                .enumerate()
                .min_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Greater))
                .unwrap_or_else(|| unreachable!("Cannot find the minimum distance"));
            child_stacks[pole_index].push(instance);
        }

        child_stacks
    }

    /// Parallelized version of the `partition` method.
    #[must_use]
    fn par_partition<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        mut self,
        data: &D,
        mut indices: Vec<usize>,
        criteria: &C,
        seed: Option<u64>,
    ) -> Self {
        indices = if !self.is_singleton() && criteria(&self) {
            let (extrema, mut indices, extremal_distances) = self.par_find_extrema(data);
            let child_stacks = self.par_split_by_extrema(data, extrema, indices);
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
            let children = Children::new(children, arg_extrema, extremal_distances);
            self = self.set_children(children);
            indices
        } else {
            indices
        };
        self.set_indices(indices);

        self
    }
}
