//! CAKES search with a single shard.

use core::cmp::Ordering;

use distances::Number;
use rayon::prelude::*;

use crate::{knn, rnn, Cluster, Dataset, Instance, PartitionCriteria, Tree};

use super::Search;

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
pub struct SingleShard<I: Instance, U: Number, D: Dataset<I, U>> {
    /// The tree used for the search.
    tree: Tree<I, U, D>,
    /// Best rnn-search algorithm.
    best_rnn: Option<rnn::Algorithm>,
    /// Best knn-search algorithm.
    best_knn: Option<knn::Algorithm>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> SingleShard<I, U, D> {
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
            best_rnn: None,
            best_knn: None,
        }
    }

    /// Returns a reference to the dataset.
    pub const fn data(&self) -> &D {
        self.tree.data()
    }

    /// Returns a reference to the tree.
    pub const fn tree(&self) -> &Tree<I, U, D> {
        &self.tree
    }

    /// A helper function for sampling query indices for tuning.
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth in the tree to sample at.
    ///
    /// # Returns
    ///
    /// A vector of indices of cluster centers at the given depth.
    fn sample_query_indices(&self, depth: usize) -> Vec<usize> {
        self.tree
            .root()
            .subtree()
            .into_iter()
            .filter(|&c| c.depth() == depth || c.is_leaf() && c.depth() < depth)
            .map(Cluster::arg_center)
            .collect()
    }
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Search<I, U, D> for SingleShard<I, U, D> {
    fn num_shards(&self) -> usize {
        1
    }

    fn shard_cardinalities(&self) -> Vec<usize> {
        vec![self.tree.data().cardinality()]
    }

    fn auto_tune_rnn(&mut self, radius: U, tuning_depth: usize) {
        let queries = self
            .sample_query_indices(tuning_depth)
            .into_iter()
            .map(|i| &self.data()[i])
            .collect::<Vec<_>>();

        (self.best_rnn, _, _) = rnn::Algorithm::variants()
            .iter()
            .map(|&algo| {
                let start = std::time::Instant::now();
                let hits = queries
                    .par_iter()
                    .map(|query| self.rnn_search(query, radius, algo))
                    .collect::<Vec<_>>();
                let elapsed = start.elapsed().as_secs_f32();
                (Some(algo), hits, elapsed)
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|| unreachable!("There are several variants of rnn-search."));
    }

    fn tuned_rnn_algorithm(&self) -> rnn::Algorithm {
        self.best_rnn.unwrap_or_default()
    }

    fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)> {
        algo.search(query, radius, &self.tree)
    }

    fn linear_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.rnn_search(query, radius, rnn::Algorithm::Linear)
    }

    fn auto_tune_knn(&mut self, k: usize, tuning_depth: usize) {
        let queries = self
            .sample_query_indices(tuning_depth)
            .into_iter()
            .map(|i| &self.data()[i])
            .collect::<Vec<_>>();

        (self.best_knn, _, _) = knn::Algorithm::variants()
            .iter()
            .map(|&algo| {
                let start = std::time::Instant::now();
                let hits = queries
                    .par_iter()
                    .map(|query| self.knn_search(query, k, algo))
                    .collect::<Vec<_>>();
                let elapsed = start.elapsed().as_secs_f32();
                (Some(algo), hits, elapsed)
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|| unreachable!("There are several variants of knn-search."));
    }

    fn tuned_knn_algorithm(&self) -> knn::Algorithm {
        self.best_knn.unwrap_or_default()
    }

    fn knn_search(&self, query: &I, k: usize, algo: knn::Algorithm) -> Vec<(usize, U)> {
        algo.search(&self.tree, query, k)
    }

    fn linear_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        self.knn_search(query, k, knn::Algorithm::Linear)
    }
}
