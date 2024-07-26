//! The various ways to store the indices of a `Cluster`.

use distances::Number;
use rayon::prelude::*;

use crate::{Dataset, ParDataset};

use super::Cluster;

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
    /// Returns a new `IndexStore` that stores the indices of every `Cluster`.
    pub const fn new_every_cluster(indices: Vec<usize>) -> Self {
        Self::EveryCluster(indices)
    }

    /// Returns a new `IndexStore` that stores the indices of only the leaf
    /// `Cluster`s.
    pub const fn new_leaf_only(indices: Option<Vec<usize>>) -> Self {
        Self::LeafOnly(indices)
    }

    /// Returns a new `IndexStore` that stores the indices of the instances in a
    /// `Cluster` after a permutation of the dataset.
    pub const fn new_post_permutation(offset: usize) -> Self {
        Self::PostPermutation(offset)
    }

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

    /// Repeat the given distances with all the indices of the instances in the `Cluster`.
    pub fn repeat_distance<U: Number, C: Cluster<U>>(&self, c: &C, d: U) -> Vec<(usize, U)> {
        match self {
            Self::EveryCluster(indices) => indices.iter().map(|&i| (i, d)).collect(),
            Self::LeafOnly(indices) => c.children().map_or_else(
                || {
                    indices
                        .clone()
                        .unwrap_or_else(|| {
                            unreachable!("Cannot repeat the distance with the indices of the instances in a leaf")
                        })
                        .iter()
                        .map(|&i| (i, d))
                        .collect()
                },
                |children| {
                    children
                        .clusters()
                        .iter()
                        .flat_map(|&c| self.repeat_distance(c, d))
                        .collect()
                },
            ),
            Self::PostPermutation(offset) => ((*offset)..((*offset) + c.cardinality())).map(|i| (i, d)).collect(),
        }
    }

    /// Returns the distances between a given `query` and the instances in the `Cluster`.
    pub fn distances<I, U: Number, D: Dataset<I, U>, C: Cluster<U>>(
        &self,
        data: &D,
        query: &I,
        c: &C,
    ) -> Vec<(usize, U)> {
        match self {
            Self::EveryCluster(indices) => indices.iter().map(|&i| (i, data.query_to_one(query, i))).collect(),
            Self::LeafOnly(indices) => c.children().map_or_else(
                || {
                    indices
                        .clone()
                        .unwrap_or_else(|| {
                            unreachable!("Cannot find the distances between a query and the instances in a leaf")
                        })
                        .iter()
                        .map(|&i| (i, data.query_to_one(query, i)))
                        .collect()
                },
                |children| {
                    children
                        .clusters()
                        .iter()
                        .flat_map(|&c| self.distances(data, query, c))
                        .collect()
                },
            ),
            Self::PostPermutation(offset) => {
                let range = (*offset)..(*offset + c.cardinality());
                range.map(|i| (i, data.query_to_one(query, i))).collect()
            }
        }
    }

    /// Parallel version of `distances`.
    pub fn par_distances<I, U, D, C>(&self, data: &D, query: &I, c: &C) -> Vec<(usize, U)>
    where
        I: Send + Sync,
        U: Number,
        D: ParDataset<I, U>,
        C: Cluster<U> + Send + Sync,
    {
        match self {
            Self::EveryCluster(indices) => indices.par_iter().map(|&i| (i, data.query_to_one(query, i))).collect(),
            Self::LeafOnly(indices) => c.children().map_or_else(
                || {
                    indices
                        .clone()
                        .unwrap_or_else(|| {
                            unreachable!("Cannot find the distances between a query and the instances in a leaf")
                        })
                        .par_iter()
                        .map(|&i| (i, data.query_to_one(query, i)))
                        .collect()
                },
                |children| {
                    children
                        .clusters()
                        .par_iter()
                        .flat_map(|&c| self.par_distances(data, query, c))
                        .collect()
                },
            ),
            Self::PostPermutation(offset) => {
                let range = (*offset)..(*offset + c.cardinality());
                range
                    .into_par_iter()
                    .map(|i| (i, data.query_to_one(query, i)))
                    .collect()
            }
        }
    }
}
