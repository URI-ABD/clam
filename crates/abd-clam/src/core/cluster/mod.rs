//! A `Cluster` is a collection of "similar" instances in a dataset.

mod ball;
mod children;
mod lfd;

use core::fmt::Debug;

use distances::Number;
use rayon::prelude::*;

use super::{Dataset, MetricSpace, ParDataset, Permutable};

pub use ball::Ball;
pub use children::Children;
pub use lfd::LFD;

// TODO: Add trait for `Cluster` adaptors.

/// The various ways to store the indices of a `Cluster`.
#[non_exhaustive]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IndexStore {
    /// Every `Cluster` stores the indices of its instances.
    EveryCluster(Vec<usize>),
    /// Only the leaf `Cluster`s store the indices of their instances.
    LeafOnly(Option<Vec<usize>>),
    /// The dataset has been reordered and the indices are stored as an offset.
    PostPermutation(usize),
}

impl IndexStore {
    /// Returns the indices of the instances in the `Cluster`.
    pub fn indices<U: Number, C: Cluster<U>>(&self, c: &C) -> Vec<usize> {
        match self {
            Self::EveryCluster(indices) => indices.clone(),
            Self::LeafOnly(indices) => c.children().map_or_else(
                || {
                    indices
                        .clone()
                        .unwrap_or_else(|| unreachable!("Cannot find the indices of the instances in a leaf"))
                },
                |children| children.clusters().iter().flat_map(|&c| self.indices(c)).collect(),
            ),
            Self::PostPermutation(offset) => ((*offset)..((*offset) + c.cardinality())).collect(),
        }
    }
}

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

    /// Computes the distance from the `Cluster`'s center to a given instance.
    fn distance_to_instance<I, D: Dataset<I, U>>(&self, data: &D, instance: &I) -> U {
        let center = data.get(self.arg_center());
        MetricSpace::one_to_one(data, center, instance)
    }

    /// Computes the distance from the `Cluster`'s center to another `Cluster`'s center.
    fn distance_to_other<I, D: Dataset<I, U>>(&self, data: &D, other: &Self) -> U {
        Dataset::one_to_one(data, self.arg_center(), other.arg_center())
    }

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

    /// Gets the indices of the instances in the `Cluster`, assuming that
    /// the `IndexStore` is `EveryCluster`.
    ///
    /// # Panics
    ///
    /// Panics if the `IndexStore` is not `EveryCluster`.
    #[allow(clippy::panic)]
    fn indices_every_cluster(&self) -> &[usize] {
        match self.index_store() {
            IndexStore::EveryCluster(indices) => indices,
            _ => panic!("Should not call `indices_every_cluster` on a Cluster without `IndexStore::EveryCluster`"),
        }
    }

    /// Gets the indices of the instances in the `Cluster`, assuming that
    /// the `IndexStore` is `LeafOnly`.
    ///
    /// # Panics
    ///
    /// Panics if the `IndexStore` is not `LeafOnly`.
    #[allow(clippy::panic)]
    fn indices_leaf_only(&self) -> Vec<usize>
    where
        Self: Sized,
    {
        match self.index_store() {
            IndexStore::LeafOnly(indices) => self.children().map_or_else(
                || {
                    indices
                        .clone()
                        .unwrap_or_else(|| unreachable!("Cannot find the indices of the instances in a leaf"))
                },
                |children| {
                    children
                        .clusters()
                        .iter()
                        .flat_map(|&c| c.indices_leaf_only())
                        .collect()
                },
            ),
            _ => panic!("Should not call `indices_leaf_only` on a Cluster without `IndexStore::LeafOnly`"),
        }
    }

    /// Gets the indices of the instances in the `Cluster`, assuming that
    /// the `IndexStore` is `PostPermutation`.
    ///
    /// # Panics
    ///
    /// Panics if the `IndexStore` is not `PostPermutation`.
    #[allow(clippy::panic)]
    fn indices_post_permutation(&self) -> core::ops::Range<usize> {
        match self.index_store() {
            &IndexStore::PostPermutation(offset) => offset..(offset + self.cardinality()),
            _ => panic!("Should not call `indices_post_permutation` on a Cluster without `IndexStore::PostPermutation`"),
        }
    }
}

/// `Cluster`s that can be partitioned into child `Cluster`s, and recursively partitioned into a tree.
pub trait Partition<U: Number>: Cluster<U> {
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
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::new(data, &indices, 0, seed);
        root.partition(data, indices, criteria, seed)
    }

    /// Creates a new `Cluster` tree and stores the indices at leaf `Cluster`s.
    fn new_tree_and_index_leaf<I, D: Dataset<I, U>, C: Fn(&Self) -> bool>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (mut root, indices) = Self::new_tree(data, criteria, seed);
        root.index_leaf_only();
        (root, indices)
    }

    /// Creates a new `Cluster` tree, permutes the `Dataset`, and stores the indices as an offset.
    fn new_tree_and_permute<I, D: Dataset<I, U> + Permutable, C: Fn(&Self) -> bool>(
        data: &mut D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
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
        // Find the cardinality of the `Cluster`, for checking later.
        let cardinality = instances.len() + extrema.len();

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
            // TODO: This next line looks fishy.
            child_stacks[pole_index].push(instance);
        }

        // TODO: Remove this check after testing.
        // Check that each child has at least one instance.
        for stack in &child_stacks {
            if stack.is_empty() {
                unreachable!("Cannot partition a Cluster with an empty child")
            }
        }
        // Check that the total number of instances is preserved.
        let total_instances: usize = child_stacks.iter().map(Vec::len).sum();
        if total_instances != cardinality {
            unreachable!("Partitioning a Cluster resulted in a loss of instances")
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
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
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
    fn index_every_cluster(&mut self)
    where
        Self: Sized,
    {
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
    fn index_leaf_only(&mut self)
    where
        Self: Sized,
    {
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
    fn index_post_permutation(&mut self, offset: Option<usize>)
    where
        Self: Sized,
    {
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
pub trait ParPartition<U: Number>: Partition<U> + Send + Sync {
    /// Parallelized version of the `new` method.
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize)
    where
        Self: Sized;

    /// Parallelized version of the `new_tree` method.
    fn par_new_tree<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let (root, _) = Self::par_new(data, &indices, 0, seed);
        root.par_partition(data, indices, criteria, seed)
    }

    /// Parallelized version of the `new_tree_and_index_leaf` method.
    fn par_new_tree_and_index_leaf<I: Send + Sync, D: ParDataset<I, U>, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (mut root, indices) = Self::par_new_tree(data, criteria, seed);
        root.par_index_leaf_only();
        (root, indices)
    }

    /// Parallelized version of the `new_tree_and_permute` method.
    fn par_new_tree_and_permute<I: Send + Sync, D: ParDataset<I, U> + Permutable, C: (Fn(&Self) -> bool) + Send + Sync>(
        data: &mut D,
        criteria: &C,
        seed: Option<u64>,
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (mut root, indices) = Self::par_new_tree(data, criteria, seed);
        root.par_index_post_permutation(None);
        data.permute(&indices);
        (root, indices)
    }

    /// Parallelized version of the `find_extrema` method.
    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>);

    /// Parallelized version of the `partition_once` method.
    fn par_split_by_extrema<I: Send + Sync, D: ParDataset<I, U>>(
        &self,
        data: &D,
        extrema: Vec<usize>,
        instances: Vec<usize>,
    ) -> Vec<Vec<usize>> {
        // Find the cardinality of the `Cluster`, for checking later.
        let cardinality = instances.len() + extrema.len();

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
            // TODO: This next line looks fishy.
            child_stacks[pole_index].push(instance);
        }

        // TODO: Remove this check after testing.
        // Check that each child has at least one instance.
        for stack in &child_stacks {
            if stack.is_empty() {
                unreachable!("Cannot partition a Cluster with an empty child")
            }
        }
        // Check that the total number of instances is preserved.
        let total_instances: usize = child_stacks.iter().map(Vec::len).sum();
        if total_instances != cardinality {
            unreachable!("Partitioning a Cluster resulted in a loss of instances")
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
    ) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
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
    fn par_index_every_cluster(&mut self)
    where
        Self: Sized,
    {
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
    fn par_index_leaf_only(&mut self)
    where
        Self: Sized,
    {
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
    fn par_index_post_permutation(&mut self, offset: Option<usize>)
    where
        Self: Sized,
    {
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
