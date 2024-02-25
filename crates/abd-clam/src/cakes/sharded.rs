//! CAKES search with sharded datasets.

use core::ops::AddAssign;

use distances::Number;
use rayon::prelude::*;

use super::{Search, SingleShard};
use crate::{knn, rnn, Dataset, Instance};

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
pub struct RandomlySharded<I: Instance, U: Number, D: Dataset<I, U>> {
    /// A random sample of the full dataset.
    sample_shard: SingleShard<I, U, D>,
    /// The full shards.
    shards: Vec<SingleShard<I, U, D>>,
    /// The Dataset.
    offsets: Vec<usize>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> RandomlySharded<I, U, D> {
    /// Creates a new `ShardedCakes` instance.
    ///
    /// # Arguments
    ///
    /// * `shards` - The shards to use.
    #[must_use]
    pub fn new(mut shards: Vec<SingleShard<I, U, D>>) -> Self {
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

    /// Returns the shards.
    pub fn shards(&self) -> Vec<&SingleShard<I, U, D>> {
        core::iter::once(&self.sample_shard).chain(self.shards.iter()).collect()
    }

    /// Returns the offsets of the shard indices.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Search<I, U, D> for RandomlySharded<I, U, D> {
    #[allow(clippy::similar_names)]
    fn save(&self, path: &std::path::Path) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("Path does not exist: {path:?}"));
        }

        if !path.is_dir() {
            return Err(format!("Path is not a directory: {path:?}"));
        }

        let sample_shard_dir = path.join("sample_shard");
        if !sample_shard_dir.exists() {
            std::fs::create_dir(&sample_shard_dir).map_err(|e| format!("Failed to create directory: {e:?}"))?;
        }
        self.sample_shard.save(&sample_shard_dir)?;

        let shards_dir = path.join("shards");
        if !shards_dir.exists() {
            std::fs::create_dir(&shards_dir).map_err(|e| format!("Failed to create directory: {e:?}"))?;
        }
        for (i, shard) in self.shards.iter().enumerate() {
            let shard_dir = shards_dir.join(format!("shard_{i}"));
            if !shard_dir.exists() {
                std::fs::create_dir(&shard_dir).map_err(|e| format!("Failed to create directory: {e:?}"))?;
            }
            shard.save(&shard_dir)?;
        }

        Ok(())
    }

    #[allow(clippy::similar_names)]
    fn load(path: &std::path::Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String>
    where
        Self: Sized,
    {
        if !path.exists() {
            return Err(format!("Path does not exist: {path:?}"));
        }

        if !path.is_dir() {
            return Err(format!("Path is not a directory: {path:?}"));
        }

        let sample_shard_dir = path.join("sample_shard");
        let mut shards = vec![SingleShard::load(&sample_shard_dir, metric, is_expensive)?];

        let shards_dir = path.join("shards");
        for i in 0.. {
            let shard_dir = shards_dir.join(format!("shard_{i}"));
            if !shard_dir.exists() {
                break;
            }
            let shard = SingleShard::load(&shard_dir, metric, is_expensive)?;
            shards.push(shard);
        }

        Ok(Self::new(shards))
    }

    fn num_shards(&self) -> usize {
        1 + self.shards.len()
    }

    fn shard_cardinalities(&self) -> Vec<usize> {
        core::iter::once(self.sample_shard.data().cardinality())
            .chain(self.shards.iter().map(|s| s.data().cardinality()))
            .collect()
    }

    fn tuned_rnn_algorithm(&self) -> rnn::Algorithm {
        self.sample_shard.tuned_rnn_algorithm()
    }

    fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)> {
        self.sample_shard
            .rnn_search(query, radius, algo)
            .into_par_iter()
            .chain(
                self.shards
                    .par_iter()
                    .zip(self.offsets.par_iter())
                    .map(|(shard, &o)| {
                        shard
                            .rnn_search(query, radius, algo)
                            .into_par_iter()
                            .map(move |(i, d)| (i + o, d))
                    })
                    .flatten(),
            )
            .collect()
    }

    fn linear_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.rnn_search(query, radius, rnn::Algorithm::Linear)
    }

    fn tuned_knn_algorithm(&self) -> knn::Algorithm {
        self.sample_shard.tuned_knn_algorithm()
    }

    fn knn_search(&self, query: &I, k: usize, algo: knn::Algorithm) -> Vec<(usize, U)> {
        let initial_hits = self.sample_shard.knn_search(query, k, algo);
        let mut hits_queue = knn::Hits::from_vec(k, initial_hits);

        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()) {
            let radius = hits_queue.peek();
            let new_hits = shard.rnn_search(query, radius, rnn::Algorithm::Clustered);
            hits_queue.push_batch(new_hits.into_iter().map(|(i, d)| (i + o, d)));
        }

        hits_queue.extract()
    }

    fn auto_tune_rnn(&mut self, radius: U, tuning_depth: usize) {
        self.sample_shard.auto_tune_rnn(radius, tuning_depth);
    }

    fn auto_tune_knn(&mut self, k: usize, tuning_depth: usize) {
        self.sample_shard.auto_tune_knn(k, tuning_depth);
    }

    fn linear_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let initial_hits = self.sample_shard.knn_search(query, k, knn::Algorithm::Linear);
        let mut hits_queue = knn::Hits::from_vec(k, initial_hits);

        for (shard, &o) in self.shards.iter().zip(self.offsets.iter()) {
            let new_hits = shard.knn_search(query, k, knn::Algorithm::Linear);
            hits_queue.push_batch(new_hits.into_iter().map(|(i, d)| (i + o, d)));
        }

        hits_queue.extract()
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use rand::prelude::*;
    use symagen::random_data;

    use crate::{
        cakes::{Search, SingleShard},
        knn, rnn, Dataset, PartitionCriteria, VecDataset,
    };

    use super::RandomlySharded;

    fn metric(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
        distances::vectors::euclidean(a, b)
    }

    #[test]
    fn vectors() {
        let seed = 42;
        let (cardinality, dimensionality) = (10_000, 10);
        let (min_val, max_val) = (-1., 1.);

        let data_vec = random_data::random_tabular(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(42),
        );

        let num_queries = 100;
        let queries = random_data::random_tabular(
            num_queries,
            dimensionality,
            min_val,
            max_val,
            &mut rand::rngs::StdRng::seed_from_u64(43),
        );

        let name = format!("test-full");
        let data = VecDataset::new(name, data_vec.clone(), metric, false);
        let criteria = PartitionCriteria::default();
        let cakes = SingleShard::new(data, Some(seed), &criteria);

        let num_shards = 10;
        let max_cardinality = cardinality / num_shards;
        let name = format!("test-sharded");
        let data_shards = VecDataset::new(name, data_vec, metric, false).make_shards(max_cardinality);
        let shards = data_shards
            .into_iter()
            .map(|d| SingleShard::new(d, Some(seed), &criteria))
            .collect::<Vec<_>>();
        let sharded_cakes = {
            let mut cakes = RandomlySharded::new(shards);
            cakes.auto_tune_knn(10, 7);
            cakes
        };

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

        for k in [100, 10, 1] {
            for (i, query) in queries.iter().enumerate() {
                let cakes_hits = {
                    let mut hits = cakes.knn_search(query, k, knn::Algorithm::Linear);
                    hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
                    hits
                };

                let sharded_hits = {
                    let mut hits = sharded_cakes.tuned_knn_search(query, k);
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
