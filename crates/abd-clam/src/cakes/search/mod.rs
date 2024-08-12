//! Entropy scaling search algorithms.

mod knn_breadth_first;
mod knn_depth_first;
mod knn_repeated_rnn;
mod rnn_clustered;

use distances::Number;
use rayon::prelude::*;

use crate::{
    cluster::ParCluster,
    dataset::ParDataset,
    linear_search::{LinearSearch, ParLinearSearch},
    Cluster, Dataset,
};

/// The different algorithms that can be used for search.
///
/// - `RnnClustered` - Ranged Nearest Neighbors search using the tree.
/// - `KnnRepeatedRnn` - K-Nearest Neighbors search using repeated `RnnClustered` searches.
/// - `KnnBreadthFirst` - K-Nearest Neighbors search using a breadth-first sieve.
/// - `KnnDepthFirst` - K-Nearest Neighbors search using a depth-first sieve.
///
/// See the `CAKES` paper for more information on these algorithms.
#[derive(Clone, Copy)]
pub enum Algorithm<U: Number> {
    /// Ranged Nearest Neighbors search using the tree.
    ///
    /// # Parameters
    ///
    /// - `U` - The radius to search within.
    RnnClustered(U),
    /// K-Nearest Neighbors search using repeated `RnnClustered` searches.
    ///
    /// # Parameters
    ///
    /// - `usize` - The number of neighbors to search for.
    /// - `U` - The maximum multiplier for the radius when repeating the search.
    KnnRepeatedRnn(usize, U),
    /// K-Nearest Neighbors search using a breadth-first sieve.
    KnnBreadthFirst(usize),
    /// K-Nearest Neighbors search using a depth-first sieve.
    KnnDepthFirst(usize),
}

impl<U: Number> Algorithm<U> {
    /// Perform the search using the algorithm.
    pub fn search<I, D: Dataset<I, U>, C: Cluster<I, U, D>>(&self, data: &D, root: &C, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::search(data, root, query, *radius),
            Self::KnnRepeatedRnn(k, max_multiplier) => knn_repeated_rnn::search(data, root, query, *k, *max_multiplier),
            Self::KnnBreadthFirst(k) => knn_breadth_first::search(data, root, query, *k),
            Self::KnnDepthFirst(k) => knn_depth_first::search(data, root, query, *k),
        }
    }

    /// Parallel version of the `search` method.
    pub fn par_search<I: Send + Sync, D: ParDataset<I, U>, C: ParCluster<I, U, D>>(
        &self,
        data: &D,
        root: &C,
        query: &I,
    ) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::par_search(data, root, query, *radius),
            Self::KnnRepeatedRnn(k, max_multiplier) => {
                knn_repeated_rnn::par_search(data, root, query, *k, *max_multiplier)
            }
            Self::KnnBreadthFirst(k) => knn_breadth_first::par_search(data, root, query, *k),
            Self::KnnDepthFirst(k) => knn_depth_first::par_search(data, root, query, *k),
        }
    }

    /// Search via a linear scan.
    pub fn linear_search<I, D: LinearSearch<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.rnn(query, *radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnDepthFirst(k) | Self::KnnBreadthFirst(k) => data.knn(query, *k),
        }
    }

    /// Batched version of the `linear_search` method
    pub fn batch_linear_search<I, D: LinearSearch<I, U>>(&self, data: &D, queries: &[I]) -> Vec<Vec<(usize, U)>> {
        queries.iter().map(|query| self.linear_search(data, query)).collect()
    }

    /// Parallel version of the `linear_search` method
    pub fn par_linear_search<I: Send + Sync, D: ParLinearSearch<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.par_rnn(query, *radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnDepthFirst(k) | Self::KnnBreadthFirst(k) => data.par_knn(query, *k),
        }
    }

    /// Parallel version of the `batch_linear_search` method
    pub fn par_batch_linear_search<I: Send + Sync, D: ParLinearSearch<I, U>>(
        &self,
        data: &D,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.linear_search(data, query))
            .collect()
    }

    /// Batched version of the `par_linear_search` method
    pub fn batch_par_linear_search<I: Send + Sync, D: ParLinearSearch<I, U>>(
        &self,
        data: &D,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        queries
            .iter()
            .map(|query| self.par_linear_search(data, query))
            .collect()
    }

    /// Parallel version of the `batch_par_linear_search` method
    pub fn par_batch_par_linear_search<I: Send + Sync, D: ParLinearSearch<I, U>>(
        &self,
        data: &D,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.par_linear_search(data, query))
            .collect()
    }

    /// Get the name of the algorithm.
    pub fn name(&self) -> String {
        match self {
            Self::RnnClustered(r) => format!("RnnClustered({r})"),
            Self::KnnRepeatedRnn(k, m) => format!("KnnRepeatedRnn({k}, {m})"),
            Self::KnnBreadthFirst(k) => format!("KnnBreadthFirst({k})"),
            Self::KnnDepthFirst(k) => format!("KnnDepthFirst({k})"),
        }
    }

    /// Get the name of the variant of algorithm.
    pub const fn variant_name(&self) -> &str {
        match self {
            Self::RnnClustered(_) => "RnnClustered",
            Self::KnnRepeatedRnn(_, _) => "KnnRepeatedRnn",
            Self::KnnBreadthFirst(_) => "KnnBreadthFirst",
            Self::KnnDepthFirst(_) => "KnnDepthFirst",
        }
    }

    /// Same variant of the algorithm with different parameters.
    #[must_use]
    pub const fn with_params(&self, radius: U, k: usize) -> Self {
        match self {
            Self::RnnClustered(_) => Self::RnnClustered(radius),
            Self::KnnRepeatedRnn(_, m) => Self::KnnRepeatedRnn(k, *m),
            Self::KnnBreadthFirst(_) => Self::KnnBreadthFirst(k),
            Self::KnnDepthFirst(_) => Self::KnnDepthFirst(k),
        }
    }
}
