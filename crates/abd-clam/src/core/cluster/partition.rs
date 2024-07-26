//! Traits for partitioning a `Cluster` and making trees.

use distances::Number;
use rayon::prelude::*;

use crate::{Dataset, ParDataset, Permutable};

use super::{Children, Cluster, IndexStore, ParCluster};

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively partitioned into a tree.
pub trait Partition<U: Number>: Cluster<U> + Sized {
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
    /// - The instances in the `Cluster` in depth-first order of traversal of
    ///  the tree.
    fn new_tree<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::new(data, &indices, 0, seed);
        root.partition(data, indices, criteria, seed)
    }

    /// Creates a new `Cluster` tree and stores the indices at leaf `Cluster`s.
    fn new_tree_and_index_leaf<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let (mut root, indices) = Self::new_tree(data, criteria, seed);
        root.index_leaf_only();
        (root, indices)
    }

    /// Creates a new `Cluster` tree, permutes the `Dataset`, and stores the indices as an offset.
    fn new_tree_and_permute<I, D: Dataset<I, U> + Permutable, C: Fn(&Self) -> bool>(
        data: &mut D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let (mut root, indices) = Self::new_tree(data, criteria, seed);
        root.index_post_permutation(None);
        data.permute(&indices);
        (root, indices)
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
    fn partition<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(
        mut self,
        data: &D,
        mut indices: Vec<usize>,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        indices = if !self.is_singleton() && criteria(&self) {
            let (extrema, mut indices, extremal_distances) = self.find_extrema(data);
            let child_stacks = self.split_by_extrema(data, extrema, indices);
            let depth = self.depth() + 1;
            let (children, other) = child_stacks
                .into_iter()
                .map(|mut child_indices| {
                    let (mut child, arg_r) = Self::new(data, &child_indices, depth, seed);
                    (child, child_indices) = child.partition(data, child_indices, criteria, seed);
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
        self.set_index_store(IndexStore::EveryCluster(indices.clone()));

        (self, indices)
    }

    /// Changes the `IndexStore` of all `Cluster`s in the tree to `IndexStore::EveryCluster`.
    fn index_every_cluster(&mut self) {
        let indices = match self.index_store() {
            IndexStore::EveryCluster(indices) => indices.clone(),
            #[allow(clippy::option_if_let_else)]
            IndexStore::LeafOnly(_) => match self.children_mut() {
                Some(children) => {
                    children.clusters_mut().into_iter().for_each(Self::index_every_cluster);
                    children
                        .clusters()
                        .iter()
                        .flat_map(|&c| c.index_store().indices(c))
                        .collect()
                }
                None => self.index_store().indices(self),
            },
            IndexStore::PostPermutation(offset) => ((*offset)..((*offset) + self.cardinality())).collect(),
        };
        self.set_index_store(IndexStore::EveryCluster(indices));
    }

    /// Changes the `IndexStore` of all `Cluster`s in the tree to `IndexStore::LeafOnly`.
    fn index_leaf_only(&mut self) {
        let indices = match self.index_store() {
            IndexStore::LeafOnly(indices) => indices.clone(),
            #[allow(clippy::option_if_let_else)]
            _ => match self.children_mut() {
                Some(children) => {
                    children.clusters_mut().into_iter().for_each(Self::index_leaf_only);
                    None
                }
                None => Some(self.index_store().indices(self)),
            },
        };
        self.set_index_store(IndexStore::LeafOnly(indices));
    }

    /// Changes the `IndexStore` of all `Cluster`s in the tree to `IndexStore::PostPermutation`.
    ///
    /// This should only be called from the root `Cluster` with `offset` set to `None`.
    fn index_post_permutation(&mut self, offset: Option<usize>) {
        let indices = self.index_store().indices(self);
        let offset = offset.unwrap_or_default();
        let children_mut = self.children_mut().map_or_else(Vec::new, Children::clusters_mut);
        if !children_mut.is_empty() {
            let mut child_offset = offset;
            for child in children_mut {
                child.index_post_permutation(Some(child_offset));
                child_offset += child.cardinality();
            }
        }

        let arg_radial = indices
            .iter()
            .position(|&i| i == self.arg_radial())
            .unwrap_or_else(|| unreachable!("Cannot find the radial instance after permutation"));
        self.set_arg_radial(arg_radial + offset);

        let arg_center = indices
            .iter()
            .position(|&i| i == self.arg_center())
            .unwrap_or_else(|| unreachable!("Cannot find the center instance after permutation"));
        self.set_arg_center(arg_center + offset);

        self.set_index_store(IndexStore::PostPermutation(offset));
    }
}

/// `Cluster`s that use and provide parallelized methods.
#[allow(clippy::module_name_repetitions)]
pub trait ParPartition<U: Number>: ParCluster<U> + Sized {
    /// Parallelized version of the `new_tree` method.
    fn par_new_tree<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::par_new(data, &indices, 0, seed);
        root.par_partition(data, indices, criteria, seed)
    }

    /// Parallelized version of the `new_tree_and_index_leaf` method.
    fn par_new_tree_and_index_leaf<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let (mut root, indices) = Self::par_new_tree(data, criteria, seed);
        root.par_index_leaf_only();
        (root, indices)
    }

    /// Parallelized version of the `new_tree_and_permute` method.
    fn par_new_tree_and_permute<
        I: Send + Sync,
        D: ParDataset<I, U> + Permutable,
        C: (Fn(&Self) -> bool) + Send + Sync,
    >(
        data: &mut D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        let (mut root, indices) = Self::par_new_tree(data, criteria, seed);
        root.par_index_post_permutation(None);
        data.permute(&indices);
        (root, indices)
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
    fn par_partition<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        mut self,
        data: &D,
        mut indices: Vec<usize>,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>) {
        indices = if !self.is_singleton() && criteria(&self) {
            let (extrema, mut indices, extremal_distances) = self.par_find_extrema(data);
            let child_stacks = self.par_split_by_extrema(data, extrema, indices);
            let depth = self.depth() + 1;
            let (children, other) = child_stacks
                .into_par_iter()
                .map(|mut child_instances| {
                    let (mut child, arg_r) = Self::par_new(data, &child_instances, depth, seed);
                    (child, child_instances) = child.par_partition(data, child_instances, criteria, seed);
                    (child, (arg_r, child_instances))
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
        self.set_index_store(IndexStore::EveryCluster(indices.clone()));

        (self, indices)
    }

    /// Parallelized version of the `index_every_cluster` method.
    fn par_index_every_cluster(&mut self) {
        let indices = match self.index_store() {
            IndexStore::EveryCluster(indices) => indices.clone(),
            #[allow(clippy::option_if_let_else)]
            IndexStore::LeafOnly(_) => match self.children_mut() {
                Some(children) => {
                    children
                        .clusters_mut()
                        .into_par_iter()
                        .for_each(Self::par_index_every_cluster);
                    children
                        .clusters()
                        .iter()
                        .flat_map(|&c| c.index_store().indices(c))
                        .collect()
                }
                None => self.index_store().indices(self),
            },
            IndexStore::PostPermutation(offset) => ((*offset)..((*offset) + self.cardinality())).collect(),
        };
        self.set_index_store(IndexStore::EveryCluster(indices));
    }

    /// Parallelized version of the `index_leaf_only` method.
    fn par_index_leaf_only(&mut self) {
        let indices = match self.index_store() {
            IndexStore::LeafOnly(indices) => indices.clone(),
            #[allow(clippy::option_if_let_else)]
            _ => match self.children_mut() {
                Some(children) => {
                    children
                        .clusters_mut()
                        .into_par_iter()
                        .for_each(Self::par_index_leaf_only);
                    None
                }
                None => Some(self.index_store().indices(self)),
            },
        };
        self.set_index_store(IndexStore::LeafOnly(indices));
    }

    /// Parallelized version of the `index_post_permutation` method.
    fn par_index_post_permutation(&mut self, offset: Option<usize>) {
        let indices = self.index_store().indices(self);
        let offset = offset.unwrap_or_default();
        let children_mut = self.children_mut().map_or_else(Vec::new, Children::clusters_mut);
        if !children_mut.is_empty() {
            let mut child_offset = offset;
            for child in children_mut {
                child.par_index_post_permutation(Some(child_offset));
                child_offset += child.cardinality();
            }
        }

        let arg_radial = indices
            .iter()
            .position(|&i| i == self.arg_radial())
            .unwrap_or_else(|| unreachable!("Cannot find the radial instance after permutation"));
        self.set_arg_radial(arg_radial + offset);

        let arg_center = indices
            .iter()
            .position(|&i| i == self.arg_center())
            .unwrap_or_else(|| unreachable!("Cannot find the center instance after permutation"));
        self.set_arg_center(arg_center + offset);

        self.set_index_store(IndexStore::PostPermutation(offset));
    }
}
