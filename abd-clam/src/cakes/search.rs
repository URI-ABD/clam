//! Supplies the `Search` trait.

use distances::Number;

use crate::{knn, rnn, Dataset, Instance};

/// A trait for performing RNN- and KNN-Search.
pub trait Search<I: Instance, U: Number, D: Dataset<I, U>>: Send + Sync {
    /// Returns the number of shards.
    fn num_shards(&self) -> usize;

    /// Returns the cardinalities of the shards.
    fn shard_cardinalities(&self) -> Vec<usize>;

    /// Returns the best RNN-Search algorithm.
    ///
    /// If the algorithm has not been tuned, this will return the default variant.
    fn tuned_rnn_algorithm(&self) -> rnn::Algorithm;

    /// Performs an RNN-Search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `radius` - The radius to use for the search.
    /// * `algo` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples containing the index of the instance and its
    /// distance to the query.
    fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)>;

    /// Performs RNN-Search using the naive linear algorithm.
    fn linear_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)>;

    /// Returns the best KNN-Search algorithm.
    ///
    /// If the algorithm has not been tuned, this will return the default variant.
    fn tuned_knn_algorithm(&self) -> knn::Algorithm;

    /// Performs a KNN-Search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `k` - The number of neighbors to search for.
    /// * `algo` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples containing the index of the instance and its
    /// distance to the query.
    fn knn_search(&self, query: &I, k: usize, algo: knn::Algorithm) -> Vec<(usize, U)>;

    /// Auto-tunes the RNN-Search algorithm and sets it as the best.
    ///
    /// # Arguments
    ///
    /// * `radius` - The radius to tune for.
    /// * `tuning_depth` - The depth to use for tuning.
    fn auto_tune_rnn(&mut self, radius: U, tuning_depth: usize);

    /// Auto-tunes the KNN-Search algorithm and sets it as the best.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of neighbors to tune for.
    /// * `tuning_depth` - The depth to use for tuning.
    fn auto_tune_knn(&mut self, k: usize, tuning_depth: usize);

    /// Performs KNN-Search using the naive linear algorithm.
    fn linear_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)>;

    /// Performs RNN-Search using the best algorithm.
    fn tuned_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        let algo = self.tuned_rnn_algorithm();
        self.rnn_search(query, radius, algo)
    }

    /// Performs KNN-Search using the best algorithm.
    fn tuned_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let algo = self.tuned_knn_algorithm();
        self.knn_search(query, k, algo)
    }
}
