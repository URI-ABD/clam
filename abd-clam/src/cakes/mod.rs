//! CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.

use core::ops::Index;

pub mod knn;
pub mod rnn;
mod search;
mod sharded;
mod singular;

use distances::Number;
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
    pub fn new_single_shard(data: D, seed: Option<u64>, criteria: &PartitionCriteria<U>) -> Self {
        Self::SingleShard(SingleShard::new(data, seed, criteria))
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
    /// A vector of tuples containing the index of the instance and the distance to the query.
    pub fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)> {
        match self {
            Self::SingleShard(ss) => ss.rnn_search(query, radius, algo),
            Self::RandomlySharded(rs) => rs.rnn_search(query, radius, algo),
        }
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
    /// A vector of tuples containing the index of the instance and the distance to the query.
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

                rs.shards()[i].data().index(index)
            }
        }
    }
}
