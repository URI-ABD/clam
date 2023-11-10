//! CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.

use core::ops::Index;

pub mod knn;
pub mod rnn;
mod search;
mod sharded;
mod singular;

use distances::Number;
use rayon::prelude::*;
use search::Search;
use sharded::RandomlySharded;
use singular::SingleShard;

use crate::{Dataset, Instance, PartitionCriteria};

/// CAKES search.
pub enum Cakes<I: Instance, U: Number, D: Dataset<I, U>> {
    /// Search with a single shard.
    SingleShard(SingleShard<I, U, D>),
    /// Search with multiple shards.
    RandomlySharded(RandomlySharded<I, U, D>),
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Cakes<I, U, D> {
    /// Creates a new CAKES instance with a single shard dataset.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the tree.
    pub fn new(data: D, seed: Option<u64>, criteria: &PartitionCriteria<U>) -> Self {
        Self::SingleShard(SingleShard::new(data, seed, criteria))
    }

    /// Returns the references to the shard(s) of the dataset.
    pub fn shards(&self) -> Vec<&D> {
        match self {
            Self::SingleShard(ss) => vec![ss.data()],
            Self::RandomlySharded(rs) => rs.shards().iter().map(SingleShard::data).collect(),
        }
    }

    /// Creates a new CAKES instance with a randomly sharded dataset.
    ///
    /// # Arguments
    ///
    /// * `shards` - The shards of the dataset to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the tree.
    #[must_use]
    pub fn new_randomly_sharded(shards: Vec<D>, seed: Option<u64>, criteria: &PartitionCriteria<U>) -> Self {
        let shards = shards
            .into_iter()
            .map(|d| SingleShard::new(d, seed, criteria))
            .collect::<Vec<_>>();
        Self::RandomlySharded(RandomlySharded::new(shards))
    }

    /// Returns the number of shards in the dataset.
    pub fn num_shards(&self) -> usize {
        match self {
            Self::SingleShard(_) => 1,
            Self::RandomlySharded(rs) => rs.num_shards(),
        }
    }

    /// Returns the cardinalities of the shards in the dataset.
    pub fn shard_cardinalities(&self) -> Vec<usize> {
        match self {
            Self::SingleShard(ss) => ss.shard_cardinalities(),
            Self::RandomlySharded(rs) => rs.shard_cardinalities(),
        }
    }

    /// Returns the tuned RNN algorithm.
    pub fn tuned_rnn_algorithm(&self) -> rnn::Algorithm {
        match self {
            Self::SingleShard(ss) => ss.tuned_rnn_algorithm(),
            Self::RandomlySharded(rs) => rs.tuned_rnn_algorithm(),
        }
    }

    /// Performs RNN search on a batch of queries with the given algorithm.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `radius` - The search radius.
    /// * `algo` - The algorithm to use.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_rnn_search(&self, queries: &[&I], radius: U, algo: rnn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.rnn_search(q, radius, algo)).collect()
    }

    /// Performs an RNN search with the given algorithm.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `radius` - The search radius.
    /// * `algo` - The algorithm to use.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance
    /// to the query.
    pub fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)> {
        match self {
            Self::SingleShard(ss) => ss.rnn_search(query, radius, algo),
            Self::RandomlySharded(rs) => rs.rnn_search(query, radius, algo),
        }
    }

    /// Performs Linear RNN search on a batch of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `radius` - The search radius.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_linear_rnn_search(&self, queries: &[&I], radius: U) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.linear_rnn_search(q, radius)).collect()
    }

    /// Performs a linear RNN search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `radius` - The search radius.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance
    /// to the query.
    pub fn linear_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        match self {
            Self::SingleShard(ss) => ss.linear_rnn_search(query, radius),
            Self::RandomlySharded(rs) => rs.linear_rnn_search(query, radius),
        }
    }

    /// Returns the tuned KNN algorithm.
    pub fn tuned_knn_algorithm(&self) -> knn::Algorithm {
        match self {
            Self::SingleShard(ss) => ss.tuned_knn_algorithm(),
            Self::RandomlySharded(rs) => rs.tuned_knn_algorithm(),
        }
    }

    /// Performs KNN search on a batch of queries with the given algorithm.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `k` - The number of nearest neighbors to return.
    /// * `algo` - The algorithm to use.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_knn_search(&self, queries: &[&I], k: usize, algo: knn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.knn_search(q, k, algo)).collect()
    }

    /// Performs a KNN search with the given algorithm.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `k` - The number of nearest neighbors to return.
    /// * `algo` - The algorithm to use.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance to the query.
    pub fn knn_search(&self, query: &I, k: usize, algo: knn::Algorithm) -> Vec<(usize, U)> {
        match self {
            Self::SingleShard(ss) => ss.knn_search(query, k, algo),
            Self::RandomlySharded(rs) => rs.knn_search(query, k, algo),
        }
    }

    /// Automatically finds the best RNN algorithm to use.
    ///
    /// # Arguments
    ///
    /// * `radius` - The search radius.
    /// * `tuning_depth` - The number of instances to use for tuning.
    pub fn auto_tune_rnn(&mut self, radius: U, tuning_depth: usize) {
        match self {
            Self::SingleShard(ss) => ss.auto_tune_rnn(radius, tuning_depth),
            Self::RandomlySharded(rs) => rs.auto_tune_rnn(radius, tuning_depth),
        }
    }

    /// Automatically finds the best KNN algorithm to use.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of nearest neighbors to return.
    /// * `tuning_depth` - The number of instances to use for tuning.
    pub fn auto_tune_knn(&mut self, k: usize, tuning_depth: usize) {
        match self {
            Self::SingleShard(ss) => ss.auto_tune_knn(k, tuning_depth),
            Self::RandomlySharded(rs) => rs.auto_tune_knn(k, tuning_depth),
        }
    }

    /// Performs Linear KNN search on a batch of queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_linear_knn_search(&self, queries: &[&I], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.linear_knn_search(q, k)).collect()
    }

    /// Performs a linear KNN search.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance to the query.
    pub fn linear_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        match self {
            Self::SingleShard(ss) => ss.linear_knn_search(query, k),
            Self::RandomlySharded(rs) => rs.linear_knn_search(query, k),
        }
    }

    /// Performs RNN search on a batch of queries with the tuned algorithm.
    ///
    /// If the algorithm has not been tuned, this will use the default algorithm.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `radius` - The search radius.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_tuned_rnn_search(&self, queries: &[&I], radius: U) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.tuned_rnn_search(q, radius)).collect()
    }

    /// Performs a RNN search with the tuned algorithm.
    ///
    /// If the algorithm has not been tuned, this will use the default algorithm.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `radius` - The search radius.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance to the query.
    pub fn tuned_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        let algo = self.tuned_rnn_algorithm();
        self.rnn_search(query, radius, algo)
    }

    /// Performs KNN search on a batch of queries with the tuned algorithm.
    ///
    /// If the algorithm has not been tuned, this will use the default algorithm.
    ///
    /// # Arguments
    ///
    /// * `queries` - The queries to search.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// A vector of vectors of tuples containing the index of the instance and
    /// the distance to the query.
    pub fn batch_tuned_knn_search(&self, queries: &[&I], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|q| self.tuned_knn_search(q, k)).collect()
    }

    /// Performs a KNN search with the tuned algorithm.
    ///
    /// If the algorithm has not been tuned, this will use the default algorithm.
    ///
    /// # Arguments
    ///
    /// * `query` - The query instance.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// # Returns
    ///
    /// A vector of tuples containing the index of the instance and the distance to the query.
    pub fn tuned_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let algo = self.tuned_knn_algorithm();
        self.knn_search(query, k, algo)
    }
}

impl<I, U, D> Index<usize> for Cakes<I, U, D>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    type Output = I;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Self::SingleShard(ss) => ss.data().index(index),
            Self::RandomlySharded(rs) => {
                let i = rs
                    .offsets()
                    .iter()
                    .enumerate()
                    .find(|(_, &o)| o > index)
                    .map_or_else(|| rs.num_shards() - 1, |(i, _)| i - 1);

                let index = index - rs.offsets()[i];
                rs.shards()[i].data().index(index)
            }
        }
    }
}
