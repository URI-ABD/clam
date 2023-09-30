//! CAKES search with sharded datasets.

use core::ops::AddAssign;

use distances::Number;
use rayon::prelude::*;

use crate::{knn, rnn, Cakes, Dataset};

/// Cakes search with sharded datasets.
///
/// This is a wrapper around `Cakes` that allows for sharding the dataset.
///
/// # Type parameters
///
/// - `T`: The type of the dataset elements.
/// - `U`: The type of the distance values.
/// - `D`: The type of the dataset.
#[derive(Debug)]
#[allow(clippy::module_name_repetitions)]
pub struct ShardedCakes<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    /// A random sample of the full dataset.
    pub(crate) sample_shard: Cakes<T, U, D>,
    /// The full shards.
    pub(crate) shards: Vec<Cakes<T, U, D>>,
    /// The Dataset.
    pub(crate) offsets: Vec<usize>,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> ShardedCakes<T, U, D> {
    /// Creates a new `ShardedCakes` instance.
    ///
    /// # Arguments
    ///
    /// * `shards` - The shards to use.
    #[must_use]
    pub fn new(mut shards: Vec<Cakes<T, U, D>>) -> Self {
        let new_shards = shards.split_off(1);
        let sample_shard = shards
            .pop()
            .unwrap_or_else(|| unreachable!("There should be at least one shard."));

        let offsets = new_shards
            .iter()
            .scan(sample_shard.data().cardinality(), |o, d| {
                o.add_assign(d.data().cardinality());
                Some(*o)
            })
            .collect::<Vec<_>>();

        Self {
            sample_shard,
            shards: new_shards,
            offsets,
        }
    }

    /// Auto-tunes the knn-search algorithm for the sample shard.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of nearest neighbors to auto-tune for.
    /// * `sampling_depth` - The depth at which to sample for auto-tuning.
    #[must_use]
    pub fn auto_tune(mut self, k: usize, tuning_depth: usize) -> Self {
        self.sample_shard = self.sample_shard.auto_tune(k, tuning_depth);
        self
    }

    /// Returns the best knn-search algorithm for the sample shard.
    pub const fn best_knn_algorithm(&self) -> knn::Algorithm {
        self.sample_shard.best_knn
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len() + 1
    }

    /// Returns the cardinalities of the shards.
    pub fn shard_cardinalities(&self) -> Vec<usize> {
        core::iter::once(self.sample_shard.data().cardinality())
            .chain(self.shards.iter().map(|s| s.data().cardinality()))
            .collect()
    }

    /// Ranged nearest neighbor search for a batch of queries.
    pub fn batch_rnn_search(&self, queries: &[T], radius: U) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.rnn_search(query, radius))
            .collect()
    }

    /// Ranged nearest neighbor search.
    pub fn rnn_search(&self, query: T, radius: U) -> Vec<(usize, U)> {
        self.sample_shard
            .rnn_search(query, radius, rnn::Algorithm::Clustered)
            .into_par_iter()
            .chain(
                self.shards
                    .par_iter()
                    .zip(self.offsets.par_iter())
                    .map(|(shard, &o)| {
                        shard
                            .rnn_search(query, radius, rnn::Algorithm::Clustered)
                            .into_par_iter()
                            .map(move |(i, d)| (i + o, d))
                    })
                    .flatten(),
            )
            .collect()
    }

    /// K-nearest neighbor search for a batch of queries.
    pub fn batch_knn_search(&self, queries: &[T], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|&query| self.knn_search(query, k)).collect()
    }

    /// K-nearest neighbor search.
    pub fn knn_search(&self, query: T, k: usize) -> Vec<(usize, U)> {
        let hits = self.sample_shard.knn_search(query, k, self.sample_shard.best_knn);
        let mut hits = knn::Hits::from_vec(k, hits);
        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()) {
            let radius = hits.peek();
            let new_hits = shard.rnn_search(query, radius, rnn::Algorithm::Clustered);
            hits.push_batch(new_hits.into_iter().map(|(i, d)| (i + o, d)));
        }
        hits.extract()
    }

    /// Linear k-nearest neighbor search for a batch of queries.
    pub fn batch_linear_knn(&self, queries: &[T], k: usize) -> Vec<Vec<(usize, U)>> {
        queries.par_iter().map(|&query| self.linear_knn(query, k)).collect()
    }

    /// Linear k-nearest neighbor search for a query.
    pub fn linear_knn(&self, query: T, k: usize) -> Vec<(usize, U)> {
        let mut hits = knn::Hits::from_vec(k, self.sample_shard.knn_search(query, k, knn::Algorithm::Linear));
        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()) {
            let new_hits = shard.knn_search(query, k, knn::Algorithm::Linear);
            hits.push_batch(new_hits.into_iter().map(|(i, d)| (i + o, d)));
        }
        hits.extract()
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use symagen::random_data;

    use crate::{knn, rnn, Cakes, Dataset, PartitionCriteria, VecDataset};

    use super::ShardedCakes;

    fn metric(a: &[f32], b: &[f32]) -> f32 {
        distances::vectors::euclidean(a, b)
    }

    #[test]
    fn vectors() {
        let seed = 42;
        let (cardinality, dimensionality) = (10_000, 10);
        let (min_val, max_val) = (-1., 1.);

        let data_vec = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
        let data_vec = data_vec.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let num_queries = 100;
        let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);

        let name = format!("test-full");
        let data = VecDataset::new(name, data_vec.clone(), metric, false);
        let cakes = Cakes::new(data, Some(seed), PartitionCriteria::default());

        let num_shards = 10;
        let max_cardinality = cardinality / num_shards;
        let name = format!("test-sharded");
        let data_shards = VecDataset::new(name, data_vec, metric, false).make_shards(max_cardinality);
        let shards = data_shards
            .into_iter()
            .map(|d| Cakes::new(d, Some(seed), PartitionCriteria::default()))
            .collect::<Vec<_>>();
        let sharded_cakes = ShardedCakes::new(shards).auto_tune(10, 7);

        for radius in [0.0, 0.05, 0.1, 0.25, 0.5] {
            for (i, query) in queries.iter().enumerate() {
                let cakes_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let sharded_hits = {
                    let mut hits = sharded_cakes.rnn_search(query, radius);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let cakes_distances = cakes_hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
                let sharded_distances = sharded_hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
                assert_eq!(
                    cakes_distances.len(),
                    sharded_distances.len(),
                    "Failed RNN search: query: {i}, radius: {radius}"
                );

                let differences = cakes_distances
                    .iter()
                    .zip(sharded_distances.iter())
                    .enumerate()
                    .map(|(i, (a, b))| (i, (a - b).abs()))
                    .filter(|&(_, d)| d > f32::EPSILON)
                    .collect::<Vec<_>>();

                assert!(
                    differences.is_empty(),
                    "Failed RNN search: query: {i}, radius: {radius}, differences: {differences:?}"
                );
            }
        }

        for k in [100, 10, 1] {
            for (i, query) in queries.iter().enumerate() {
                let cakes_hits = {
                    let mut hits = cakes.knn_search(query, k, knn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let sharded_hits = {
                    let mut hits = sharded_cakes.knn_search(query, k);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };
                assert_eq!(sharded_hits.len(), k, "Failed KNN search: query: {i}, k: {k}");

                let cakes_distances = cakes_hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
                let sharded_distances = sharded_hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
                assert_eq!(
                    cakes_distances.len(),
                    sharded_distances.len(),
                    "Failed KNN search: query: {i}, k: {k}"
                );

                let differences = cakes_distances
                    .iter()
                    .zip(sharded_distances.iter())
                    .enumerate()
                    .map(|(i, (a, b))| (i, (a - b).abs()))
                    .filter(|&(_, d)| d > f32::EPSILON)
                    .collect::<Vec<_>>();

                assert!(
                    differences.is_empty(),
                    "Failed KNN search: query: {i}, k: {k}, differences: {differences:?}"
                );
            }
        }
    }
}
