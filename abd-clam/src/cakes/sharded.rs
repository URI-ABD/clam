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
    /// The shards.
    pub(crate) shards: Vec<Cakes<T, U, D>>,
    /// The Dataset.
    pub(crate) offsets: Vec<usize>,
}

impl<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> ShardedCakes<T, U, D> {
    /// Creates a new `ShardedCakes` instance.
    ///
    /// # Arguments
    ///
    /// * `datasets` - The sharded datasets to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the trees.
    #[allow(clippy::needless_pass_by_value)] // clippy is wrong in this case
    #[must_use]
    pub fn new(datasets_criteria: Vec<(D, PartitionCriteria<T, U>)>, seed: Option<u64>) -> Self {
        let shards = datasets_criteria
            .into_iter()
            .map(|(data, criteria)| Cakes {
                tree: Tree::new(data, seed).partition(&criteria),
            })
            .collect::<Vec<_>>();

        let offsets = shards
            .iter()
            .scan(0, |o, d| {
                o.add_assign(d.data().cardinality());
                Some(*o)
            })
            .collect();

        Self { shards, offsets }
    }

    /// Ranged nearest neighbor search for a batch of queries.
    pub fn batch_rnn_search(&self, queries: &[T], radius: U, algorithm: rnn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.rnn_search(query, radius, algorithm))
            .collect()
    }

    /// Ranged nearest neighbor search.
    pub fn rnn_search(&self, query: T, radius: U, algorithm: rnn::Algorithm) -> Vec<(usize, U)> {
        self.shards
            .par_iter()
            .zip(self.offsets.par_iter())
            .map(|(shard, &o)| {
                shard
                    .rnn_search(query, radius, algorithm)
                    .into_par_iter()
                    .map(move |(i, d)| (i + o, d))
            })
            .flatten()
            .collect()
    }

    /// K-nearest neighbor search for a batch of queries.
    pub fn batch_knn_search(&self, queries: &[T], k: usize, algorithm: knn::Algorithm) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|&query| self.knn_search(query, k, algorithm))
            .collect()
    }

    /// K-nearest neighbor search.
    pub fn knn_search(&self, query: T, k: usize, algorithm: knn::Algorithm) -> Vec<(usize, U)> {
        let mut hits = knn::Hits::from_vec(k, self.shards[0].knn_search(query, k, algorithm));
        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()).skip(1) {
            let radius = hits.peek(U::zero);
            let new_hits = shard.rnn_search(query, radius, rnn::Algorithm::Clustered);
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

        let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let num_shards = 10;
        let shards = data
            .chunks(cardinality / num_shards)
            .enumerate()
            .map(|(i, data)| {
                let name = format!("test-shard-{}", i);
                VecDataset::new(name, data.to_vec(), metric, false)
            })
            .collect::<Vec<_>>();

        let num_queries = 100;
        let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);

        let name = format!("test-full");
        let data = VecDataset::new(name, data.clone(), metric, false);
        let cakes = Cakes::new(data, Some(seed), PartitionCriteria::default());

        let shards_criteria = shards
            .into_iter()
            .map(|s| (s, PartitionCriteria::default()))
            .collect::<Vec<_>>();
        let sharded_cakes = ShardedCakes::new(shards_criteria, Some(seed));

        for radius in [0.0, 0.05, 0.1, 0.25, 0.5] {
            for (i, query) in queries.iter().enumerate() {
                let cakes_hits = {
                    let mut hits = cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let sharded_hits = {
                    let mut hits = sharded_cakes.rnn_search(query, radius, rnn::Algorithm::Clustered);
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

        let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();

        let num_shards = 10;
        let shards = data
            .chunks(cardinality / num_shards)
            .enumerate()
            .map(|(i, data)| {
                let name = format!("test-shard-{}", i);
                VecDataset::new(name, data.to_vec(), metric, false)
            })
            .collect::<Vec<_>>();

        let num_queries = 100;
        let queries = random_data::random_f32(num_queries, dimensionality, min_val, max_val, seed + 1);

        let name = format!("test-full");
        let data = VecDataset::new(name, data.clone(), metric, false);
        let cakes = Cakes::new(data, Some(seed), PartitionCriteria::default());

        let shards_criteria = shards
            .into_iter()
            .map(|s| (s, PartitionCriteria::default()))
            .collect::<Vec<_>>();
        let sharded_cakes = ShardedCakes::new(shards_criteria, Some(seed));

        for k in [100, 10, 1] {
            for (i, query) in queries.iter().enumerate() {
                let cakes_hits = {
                    let mut hits = cakes.knn_search(query, k, knn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let sharded_hits = {
                    let mut hits = sharded_cakes.knn_search(query, k, knn::Algorithm::Linear);
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
