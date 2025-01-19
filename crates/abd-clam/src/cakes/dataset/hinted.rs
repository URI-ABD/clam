//! `Dataset`s which store extra information to improve search performance.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{KnnDepthFirst, ParSearchAlgorithm, RnnClustered, SearchAlgorithm},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric,
};

use super::{ParSearchable, Searchable};

/// An extension of the `Dataset` trait to support hinting for search.
///
/// Each hint is a tuple of the number of known neighbors and the distance to
/// the farthest known neighbor. Each item may have multiple hints, stored in a
/// map where the key is the number of known neighbors and the value is the
/// distance to the farthest known neighbor.
#[allow(clippy::module_name_repetitions)]
pub trait HintedDataset<I, T: Number, C: Cluster<T>, M: Metric<I, T>>: Searchable<I, T, C, M> + Sized {
    /// Get the search hints for a specific item by index.
    fn hints_for(&self, i: usize) -> &HashMap<usize, T>;

    /// Get the search hints for a specific item by index as mutable.
    fn hints_for_mut(&mut self, i: usize) -> &mut HashMap<usize, T>;

    /// Deletes the hints for the indexed item.
    fn clear_hints_for(&mut self, i: usize) {
        self.hints_for_mut(i).clear();
    }

    /// Deletes all hints for all items.
    fn clear_all_hints(&mut self) {
        (0..self.cardinality()).for_each(|i| self.clear_hints_for(i));
    }

    /// Add a hint for the indexed item.
    #[must_use]
    fn with_hint_for(mut self, i: usize, k: usize, d: T) -> Self {
        self.hints_for_mut(i).insert(k, d);
        self
    }

    /// Add hints from a tree.
    ///
    /// For each item in the tree, the number of known neighbors and the
    /// distance to the farthest known neighbor are added as hints.
    #[must_use]
    fn with_hints_from_tree(self, root: &C, _: &M) -> Self {
        root.subtree()
            .into_iter()
            .filter(|c| c.radius() > T::ZERO)
            .map(|c| (c.arg_center(), c.cardinality(), c.radius()))
            .fold(self, |data, (i, k, d)| data.with_hint_for(i, k, d))
    }

    /// Add hints using a search algorithm.
    ///
    /// # Arguments
    ///
    /// * `metric` - The metric to use for the search.
    /// * `root` - The root of the search tree.
    /// * `alg` - The search algorithm to use.
    /// * `q` - The index of the query item.
    #[must_use]
    fn with_hints_by_search<A: SearchAlgorithm<I, T, C, M, Self>>(self, metric: &M, root: &C, alg: A) -> Self {
        (0..self.cardinality())
            .flat_map(|i| {
                let mut hits = alg
                    .search(&self, metric, root, self.get(i))
                    .into_iter()
                    .filter(|&(_, d)| d > T::ZERO)
                    .collect::<Vec<_>>();
                hits.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                hits.into_iter().enumerate().map(move |(k, (_, d))| (i, k, d))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(self, |data, (i, k, d)| data.with_hint_for(i, k, d))
    }

    /// Add hints from a tree and several search algorithms.
    #[must_use]
    fn with_hints_from(self, metric: &M, root: &C, radius: T, k: usize) -> Self {
        self.with_hints_from_tree(root, metric)
            .with_hints_by_search(metric, root, RnnClustered(radius))
            .with_hints_by_search(metric, root, KnnDepthFirst(k))
    }
}

/// Parallel version of [`HintedDataset`](crate::cakes::dataset::hinted::HintedDataset).
#[allow(clippy::module_name_repetitions)]
pub trait ParHintedDataset<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>>:
    HintedDataset<I, T, C, M> + ParSearchable<I, T, C, M>
{
    /// Parallel version of [`HintedDataset::with_hints_by_search`](crate::cakes::dataset::hinted::HintedDataset::with_hints_by_search).
    #[must_use]
    fn par_with_hints_by_search<A: ParSearchAlgorithm<I, T, C, M, Self>>(self, metric: &M, root: &C, alg: A) -> Self {
        // todo!()
        (0..self.cardinality())
            .into_par_iter()
            .flat_map(|i| {
                let mut hits = alg
                    .par_search(&self, metric, root, self.get(i))
                    .into_par_iter()
                    .filter(|&(_, d)| d > T::ZERO)
                    .collect::<Vec<_>>();
                hits.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                hits.into_par_iter().enumerate().map(move |(k, (_, d))| (i, k, d))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .fold(self, |data, (i, k, d)| data.with_hint_for(i, k, d))
    }

    /// Parallel version of [`HintedDataset::with_hints_from`](crate::cakes::dataset::hinted::HintedDataset::with_hints_from).
    #[must_use]
    fn par_with_hints_from(self, metric: &M, root: &C, radius: T, k: usize) -> Self {
        self.with_hints_from_tree(root, metric)
            .par_with_hints_by_search(metric, root, RnnClustered(radius))
            .par_with_hints_by_search(metric, root, KnnDepthFirst(k))
    }
}
