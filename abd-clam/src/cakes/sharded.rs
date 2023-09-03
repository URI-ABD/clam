//! CAKES search with sharded datasets.

use core::ops::AddAssign;

use distances::Number;
use rayon::prelude::*;

use crate::{knn, rnn, Cakes, Dataset, PartitionCriteria, Tree};

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
    /// The algorithm to use for the sample shard.
    pub fastest_algorithm: knn::Algorithm,
    /// The full shards.
    pub(crate) shards: Vec<Cakes<T, U, D>>,
    /// The Dataset.
    pub(crate) offsets: Vec<usize>,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> ShardedCakes<T, U, D> {
    /// Creates a new `ShardedCakes` instance.
    ///
    /// Auto-tunes the knn-search algorithms for the dataset with the given `k`.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset from which to make shards and build the trees.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the trees.
    /// * `max_cardinality` - The maximum cardinality of any shard.
    /// * `k` - The number of nearest neighbors to search for.
    /// * `sampling_depth` - The depth at which to sample the shards for auto-tuning.
    #[allow(clippy::needless_pass_by_value)]
    pub fn new(
        data: D,
        seed: Option<u64>,
        criteria: PartitionCriteria<T, U>,
        max_cardinality: usize,
        k: usize,
        sampling_depth: usize,
    ) -> Self {
        let (sample_shard, shards) = {
            let mut sample_data = data.make_shards(max_cardinality);
            let datasets = sample_data.split_off(1);

            let data = sample_data
                .into_iter()
                .next()
                .unwrap_or_else(|| unreachable!("There should be at least one shard."));
            let sample_shard = Cakes {
                tree: Tree::new(data, seed).partition(&criteria),
            };

            let shards = datasets
                .into_iter()
                .map(|data| Cakes {
                    tree: Tree::new(data, seed).partition(&criteria),
                })
                .collect::<Vec<_>>();

            (sample_shard, shards)
        };

        let sample_algorithm = sample_shard.auto_tune(k, sampling_depth);

        let offsets = shards
            .iter()
            .scan(sample_shard.data().cardinality(), |o, d| {
                o.add_assign(d.data().cardinality());
                Some(*o)
            })
            .collect::<Vec<_>>();

        Self {
            sample_shard,
            fastest_algorithm: sample_algorithm,
            shards,
            offsets,
        }
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
        let mut hits = knn::Hits::from_vec(k, self.sample_shard.knn_search(query, k, self.fastest_algorithm));
        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()) {
            let radius = hits.peek(U::zero);
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

    use crate::{knn, rnn, Cakes, PartitionCriteria, VecDataset};

    use super::ShardedCakes;

    fn metric(a: &[f32], b: &[f32]) -> f32 {
        distances::vectors::euclidean(a, b)
    }

    #[test]
    fn rnn_vectors() {
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
        let name = format!("test-sharded");
        let data = VecDataset::new(name, data_vec, metric, false);
        let sharded_cakes = ShardedCakes::new(
            data,
            Some(seed),
            PartitionCriteria::default(),
            cardinality / num_shards,
            10,
            7,
        );

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
    }

    #[test]
    fn knn_vectors() {
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
        let name = format!("test-sharded");
        let data = VecDataset::new(name, data_vec, metric, false);
        let sharded_cakes = ShardedCakes::new(
            data,
            Some(seed),
            PartitionCriteria::default(),
            cardinality / num_shards,
            10,
            7,
        );

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
