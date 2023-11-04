//! CAKES search with a single shard.

use core::cmp::Ordering;

use distances::Number;
use rayon::prelude::*;

use crate::{knn, rnn, Dataset, Instance, PartitionCriteria, Tree};

/// CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.
///
/// The search time scales by the metric entropy of the dataset.
///
/// # Type Parameters
///
/// * `T` - The type of the instances.
/// * `U` - The type of the distance value.
/// * `D` - The type of the dataset.
#[derive(Debug)]
pub struct Cakes<I: Instance, U: Number, D: Dataset<I, U>> {
    /// The tree used for the search.
    pub(crate) tree: Tree<I, U, D>,
    /// Best knn-search algorithm.
    pub(crate) best_knn: Option<knn::Algorithm>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Cakes<I, U, D> {
    /// Creates a new CAKES instance.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the tree.
    pub fn new(data: D, seed: Option<u64>, criteria: &PartitionCriteria<U>) -> Self {
        Self {
            tree: Tree::new(data, seed).partition(criteria),
            best_knn: None,
        }
    }

    /// Returns a reference to the data.
    pub const fn data(&self) -> &D {
        self.tree.data()
    }

    /// Returns a reference to the tree.
    pub const fn tree(&self) -> &Tree<I, U, D> {
        &self.tree
    }

    /// Returns the depth of the tree.
    pub const fn depth(&self) -> usize {
        self.tree.depth
    }

    /// Returns the radius of the root cluster of the tree.
    pub const fn radius(&self) -> U {
        self.tree.radius()
    }

    /// Automatically tunes the algorithm to return the fastest variant for knn-search.
    #[must_use]
    pub fn auto_tune(mut self, k: usize, tuning_depth: usize) -> Self {
        let queries = self
            .tree
            .root
            .subtree()
            .into_iter()
            .filter(|&c| c.depth() == tuning_depth || c.is_leaf() && c.depth() < tuning_depth)
            .map(|c| &self.tree.data[c.arg_center()])
            .collect::<Vec<_>>();

        (self.best_knn, _, _) = knn::Algorithm::variants()
            .iter()
            .map(|&algorithm| {
                let start = std::time::Instant::now();
                let hits = self.batch_knn_search(&queries, k, algorithm);
                let elapsed = start.elapsed().as_secs_f32();
                (Some(algorithm), hits, elapsed)
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|| unreachable!("There are several variants of knn-search"));

        self
    }

    /// Performs a parallelized search for the nearest neighbors of a set of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search around.
    /// * `radius` - The radius to search within.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of vectors of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn batch_rnn_search(&self, queries: &[&I], radius: U, algorithm: rnn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.rnn_search(query, radius, algorithm))
            .collect()
    }

    /// Performs a search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `radius` - The radius to search within.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn rnn_search(&self, query: &I, radius: U, algorithm: rnn::Algorithm) -> Vec<(usize, U)> {
        algorithm.search(query, radius, &self.tree)
    }

    /// Performs a parallelized search for the nearest neighbors of a set of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of vectors of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn batch_knn_search(&self, queries: &[&I], k: usize, algorithm: knn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.knn_search(query, k, algorithm))
            .collect()
    }

    /// Performs a search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub fn knn_search(&self, query: &I, k: usize, algorithm: knn::Algorithm) -> Vec<(usize, U)> {
        algorithm.search(&self.tree, query, k)
    }

    /// Linear k-nearest neighbor search for a batch of queries.
    pub fn batch_tuned_knn(&self, queries: &[&I], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|query| self.tuned_knn(query, k)).collect()
    }

    /// Linear k-nearest neighbor search for a query.
    pub fn tuned_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        self.knn_search(query, k, self.best_knn.unwrap_or_default())
    }

    /// Linear k-nearest neighbor search for a batch of queries.
    pub fn batch_linear_knn(&self, queries: &[&I], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|query| self.linear_knn(query, k)).collect()
    }

    /// Linear k-nearest neighbor search for a query.
    pub fn linear_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        self.knn_search(query, k, knn::Algorithm::Linear)
    }
}
