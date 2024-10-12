//! Searchable dataset.

use distances::Number;
use rayon::prelude::*;

use crate::{cakes::Algorithm, cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// A dataset that can be searched with entropy-scaling algorithms.
pub trait Searchable<I, U: Number, C: Cluster<I, U, Self>>: Dataset<I, U> + Sized {
    /// Searches the dataset for the `query` instance and returns the
    /// indices of and distances to the nearest neighbors.
    fn search(&self, root: &C, query: &I, alg: Algorithm<U>) -> Vec<(usize, U)> {
        alg.search(self, root, query)
    }

    /// Batch version of the `search` method, to search for multiple queries.
    fn batch_search(&self, root: &C, queries: &[I], alg: Algorithm<U>) -> Vec<Vec<(usize, U)>> {
        queries.iter().map(|query| self.search(root, query, alg)).collect()
    }
}

/// Parallel version of the `Searchable` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParSearchable<I: Send + Sync, U: Number, C: ParCluster<I, U, Self>>:
    Searchable<I, U, C> + ParDataset<I, U>
{
    /// Parallel version of the `search` method.
    fn par_search(&self, root: &C, query: &I, alg: Algorithm<U>) -> Vec<(usize, U)> {
        alg.par_search(self, root, query)
    }

    /// Batch version of the `par_search` method.
    fn batch_par_search(&self, root: &C, queries: &[I], alg: Algorithm<U>) -> Vec<Vec<(usize, U)>> {
        queries.iter().map(|query| self.par_search(root, query, alg)).collect()
    }

    /// Parallel version of the `batch_search` method.
    fn par_batch_search(&self, root: &C, queries: &[I], alg: Algorithm<U>) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|query| self.search(root, query, alg)).collect()
    }

    /// Parallel version of the `batch_par_search` method.
    fn par_batch_par_search(&self, root: &C, queries: &[I], alg: Algorithm<U>) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.par_search(root, query, alg))
            .collect()
    }
}
