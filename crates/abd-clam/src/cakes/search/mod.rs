//! Entropy scaling search algorithms.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

mod knn_breadth_first;
mod knn_depth_first;
mod knn_repeated_rnn;
mod rnn_clustered;

/// The different algorithms that can be used for search.
///
/// - `RnnClustered` - Ranged Nearest Neighbors search using the tree.
/// - `KnnRepeatedRnn` - K-Nearest Neighbors search using repeated `RnnClustered` searches.
/// - `KnnBreadthFirst` - K-Nearest Neighbors search using a breadth-first sieve.
/// - `KnnDepthFirst` - K-Nearest Neighbors search using a depth-first sieve.
///
/// See the `CAKES` paper for more information on these algorithms.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub enum Algorithm<U: Number> {
    /// Linear RNN search.
    RnnLinear(U),
    /// Linear KNN search.
    KnnLinear(usize),
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
            Self::RnnLinear(radius) => data.rnn(query, *radius),
            Self::KnnLinear(k) => data.knn(query, *k),
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
            Self::RnnLinear(radius) => data.par_rnn(query, *radius),
            Self::KnnLinear(k) => data.par_knn(query, *k),
            Self::RnnClustered(radius) => rnn_clustered::par_search(data, root, query, *radius),
            Self::KnnRepeatedRnn(k, max_multiplier) => {
                knn_repeated_rnn::par_search(data, root, query, *k, *max_multiplier)
            }
            Self::KnnBreadthFirst(k) => knn_breadth_first::par_search(data, root, query, *k),
            Self::KnnDepthFirst(k) => knn_depth_first::par_search(data, root, query, *k),
        }
    }

    /// Batched version of the `search` method.
    pub fn batch_search<I, D: Dataset<I, U>, C: Cluster<I, U, D>>(
        &self,
        data: &D,
        root: &C,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        match self {
            Self::RnnLinear(radius) => queries.iter().map(|query| data.rnn(query, *radius)).collect(),
            Self::KnnLinear(k) => queries.iter().map(|query| data.knn(query, *k)).collect(),
            Self::RnnClustered(radius) => queries
                .iter()
                .map(|query| rnn_clustered::search(data, root, query, *radius))
                .collect(),
            Self::KnnRepeatedRnn(k, max_multiplier) => queries
                .iter()
                .map(|query| knn_repeated_rnn::search(data, root, query, *k, *max_multiplier))
                .collect(),
            Self::KnnBreadthFirst(k) => queries
                .iter()
                .map(|query| knn_breadth_first::search(data, root, query, *k))
                .collect(),
            Self::KnnDepthFirst(k) => queries
                .iter()
                .map(|query| knn_depth_first::search(data, root, query, *k))
                .collect(),
        }
    }

    /// Parallel version of the `batch_search` method.
    pub fn par_batch_search<I: Send + Sync, D: ParDataset<I, U>, C: ParCluster<I, U, D>>(
        &self,
        data: &D,
        root: &C,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        match self {
            Self::RnnLinear(radius) => queries.par_iter().map(|query| data.rnn(query, *radius)).collect(),
            Self::KnnLinear(k) => queries.par_iter().map(|query| data.knn(query, *k)).collect(),
            Self::RnnClustered(radius) => queries
                .par_iter()
                .map(|query| rnn_clustered::search(data, root, query, *radius))
                .collect(),
            Self::KnnRepeatedRnn(k, max_multiplier) => queries
                .par_iter()
                .map(|query| knn_repeated_rnn::search(data, root, query, *k, *max_multiplier))
                .collect(),
            Self::KnnBreadthFirst(k) => queries
                .par_iter()
                .map(|query| knn_breadth_first::search(data, root, query, *k))
                .collect(),
            Self::KnnDepthFirst(k) => queries
                .par_iter()
                .map(|query| knn_depth_first::search(data, root, query, *k))
                .collect(),
        }
    }

    /// Batched version of the `par_search` method.
    pub fn batch_par_search<I: Send + Sync, D: ParDataset<I, U>, C: ParCluster<I, U, D>>(
        &self,
        data: &D,
        root: &C,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        match self {
            Self::RnnLinear(radius) => queries.iter().map(|query| data.par_rnn(query, *radius)).collect(),
            Self::KnnLinear(k) => queries.iter().map(|query| data.par_knn(query, *k)).collect(),
            Self::RnnClustered(radius) => queries
                .iter()
                .map(|query| rnn_clustered::par_search(data, root, query, *radius))
                .collect(),
            Self::KnnRepeatedRnn(k, max_multiplier) => queries
                .iter()
                .map(|query| knn_repeated_rnn::par_search(data, root, query, *k, *max_multiplier))
                .collect(),
            Self::KnnBreadthFirst(k) => queries
                .iter()
                .map(|query| knn_breadth_first::par_search(data, root, query, *k))
                .collect(),
            Self::KnnDepthFirst(k) => queries
                .iter()
                .map(|query| knn_depth_first::par_search(data, root, query, *k))
                .collect(),
        }
    }

    /// Parallel version of the `batch_par_search` method.
    pub fn par_batch_par_search<I: Send + Sync, D: ParDataset<I, U>, C: ParCluster<I, U, D>>(
        &self,
        data: &D,
        root: &C,
        queries: &[I],
    ) -> Vec<Vec<(usize, U)>> {
        match self {
            Self::RnnLinear(radius) => queries.par_iter().map(|query| data.par_rnn(query, *radius)).collect(),
            Self::KnnLinear(k) => queries.par_iter().map(|query| data.par_knn(query, *k)).collect(),
            Self::RnnClustered(radius) => queries
                .par_iter()
                .map(|query| rnn_clustered::par_search(data, root, query, *radius))
                .collect(),
            Self::KnnRepeatedRnn(k, max_multiplier) => queries
                .par_iter()
                .map(|query| knn_repeated_rnn::par_search(data, root, query, *k, *max_multiplier))
                .collect(),
            Self::KnnBreadthFirst(k) => queries
                .par_iter()
                .map(|query| knn_breadth_first::par_search(data, root, query, *k))
                .collect(),
            Self::KnnDepthFirst(k) => queries
                .par_iter()
                .map(|query| knn_depth_first::par_search(data, root, query, *k))
                .collect(),
        }
    }

    /// Get the name of the algorithm.
    pub fn name(&self) -> String {
        match self {
            Self::RnnLinear(r) => format!("RnnLinear({r})"),
            Self::KnnLinear(k) => format!("KnnLinear({k})"),
            Self::RnnClustered(r) => format!("RnnClustered({r})"),
            Self::KnnRepeatedRnn(k, m) => format!("KnnRepeatedRnn({k}, {m})"),
            Self::KnnBreadthFirst(k) => format!("KnnBreadthFirst({k})"),
            Self::KnnDepthFirst(k) => format!("KnnDepthFirst({k})"),
        }
    }

    /// Get the name of the variant of algorithm.
    pub const fn variant_name(&self) -> &str {
        match self {
            Self::RnnLinear(_) => "RnnLinear",
            Self::KnnLinear(_) => "KnnLinear",
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
            Self::RnnLinear(_) => Self::RnnLinear(radius),
            Self::KnnLinear(_) => Self::KnnLinear(k),
            Self::RnnClustered(_) => Self::RnnClustered(radius),
            Self::KnnRepeatedRnn(_, m) => Self::KnnRepeatedRnn(k, *m),
            Self::KnnBreadthFirst(_) => Self::KnnBreadthFirst(k),
            Self::KnnDepthFirst(_) => Self::KnnDepthFirst(k),
        }
    }

    /// Returns the linear-search variant of the algorithm.
    #[must_use]
    pub const fn linear_variant(&self) -> Self {
        match self {
            Self::RnnClustered(r) | Self::RnnLinear(r) => Self::RnnLinear(*r),
            Self::KnnBreadthFirst(k) | Self::KnnDepthFirst(k) | Self::KnnRepeatedRnn(k, _) | Self::KnnLinear(k) => {
                Self::KnnLinear(*k)
            }
        }
    }
}
