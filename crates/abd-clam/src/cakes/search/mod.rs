//! Entropy scaling search algorithms and supporting traits.

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, metric::ParMetric, Cluster, Metric};

use super::{ParSearchable, Searchable};

mod knn_breadth_first;
mod knn_depth_first;
mod knn_hinted;
mod knn_linear;
mod knn_repeated_rnn;
mod rnn_clustered;
mod rnn_linear;

pub use knn_breadth_first::KnnBreadthFirst;
pub use knn_depth_first::KnnDepthFirst;
pub use knn_hinted::KnnHinted;
pub use knn_linear::KnnLinear;
pub use knn_repeated_rnn::KnnRepeatedRnn;
pub use rnn_clustered::RnnClustered;
pub use rnn_linear::RnnLinear;

/// Common trait for entropy scaling search algorithms.
#[allow(clippy::module_name_repetitions)]
pub trait SearchAlgorithm<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> {
    /// Return the name of the search algorithm.
    fn name(&self) -> &str;

    /// Get the radius if it is a ranged search algorithm.
    fn radius(&self) -> Option<T>;

    /// Get the value of k if it is a k-NN search algorithm.
    fn k(&self) -> Option<usize>;

    /// Perform a search using the given parameters.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `metric` - The metric to use for distance calculations.
    /// * `root` - The root of the tree to search.
    /// * `query` - The query to search around.
    ///
    /// # Returns
    ///
    /// A vector of pairs, where each pair contains the index of an item in the
    /// dataset and the distance from the query to that item.
    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)>;

    /// Batched version of `Search::search`.
    fn batch_search(&self, data: &D, metric: &M, root: &C, queries: &[I]) -> Vec<Vec<(usize, T)>> {
        queries
            .iter()
            .map(|query| self.search(data, metric, root, query))
            .collect()
    }
}

/// Parallel version of [`SearchAlgorithm`](crate::cakes::search::SearchAlgorithm).
pub trait ParSearchAlgorithm<
    I: Send + Sync,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParSearchable<I, T, C, M>,
>: SearchAlgorithm<I, T, C, M, D> + Send + Sync
{
    /// Parallel version of [`SearchAlgorithm::search`](crate::cakes::search::SearchAlgorithm::search).
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)>;

    /// Parallel version of [`SearchAlgorithm::batch_search`](crate::cakes::search::SearchAlgorithm::batch_search).
    fn par_batch_search(&self, data: &D, metric: &M, root: &C, queries: &[I]) -> Vec<Vec<(usize, T)>> {
        queries
            .par_iter()
            .map(|query| self.par_search(data, metric, root, query))
            .collect()
    }
}

/// A blanket implementation of `SearchAlgorithm` for `Box<dyn SearchAlgorithm>`.
impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for Box<dyn SearchAlgorithm<I, T, C, M, D>>
{
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn radius(&self) -> Option<T> {
        self.as_ref().radius()
    }

    fn k(&self) -> Option<usize> {
        self.as_ref().k()
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        self.as_ref().search(data, metric, root, query)
    }
}

/// A blanket implementation of `SearchAlgorithm` for `Box<dyn ParSearchAlgorithm>`.
impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    SearchAlgorithm<I, T, C, M, D> for Box<dyn ParSearchAlgorithm<I, T, C, M, D>>
{
    fn name(&self) -> &str {
        self.as_ref().name()
    }

    fn radius(&self) -> Option<T> {
        self.as_ref().radius()
    }

    fn k(&self) -> Option<usize> {
        self.as_ref().k()
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        self.as_ref().search(data, metric, root, query)
    }
}

/// A blanket implementation of `ParSearchAlgorithm` for `Box<dyn ParSearchAlgorithm>`.
impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for Box<dyn ParSearchAlgorithm<I, T, C, M, D>>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        self.as_ref().par_search(data, metric, root, query)
    }
}
