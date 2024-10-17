//! Helpers for running benchmarks on ANN datasets.

use distances::Number;
use rayon::prelude::*;

mod reader;

pub use reader::read;

/// A helper for storing training and query data for `ann-benchmarks`'s datasets
/// along with the ground truth nearest neighbors and distances.
pub struct AnnDataset<T> {
    /// The data to use for clustering.
    pub train: Vec<Vec<T>>,
    /// The queries to use for search.
    pub queries: Vec<Vec<T>>,
    /// The true neighbors of each query, given as a tuple of:
    /// * index into `train`, and
    /// * distance to the query.
    pub neighbors: Vec<Vec<(usize, f32)>>,
}

impl AnnDataset<f32> {
    /// Augment the dataset by adding noisy copies of the data.
    #[must_use]
    pub fn augment(mut self, multiplier: usize, error_rate: f32) -> Self {
        ftlog::info!("Augmenting dataset to {multiplier}x...");
        self.train = symagen::augmentation::augment_data(&self.train, multiplier, error_rate);

        self
    }

    /// Generate a random dataset with the given metric.
    #[must_use]
    pub fn gen_random(cardinality: usize, n_copies: usize, dimensionality: usize, n_queries: usize, seed: u64) -> Self {
        let train = (0..n_copies)
            .into_par_iter()
            .flat_map(|i| {
                let seed = seed + i.as_u64();
                symagen::random_data::random_tabular_seedable(cardinality, dimensionality, -1.0, 1.0, seed)
            })
            .collect::<Vec<_>>();
        let queries = symagen::random_data::random_tabular_seedable(
            n_queries,
            dimensionality,
            -1.0,
            1.0,
            seed + n_copies.as_u64(),
        );

        Self {
            train,
            queries,
            neighbors: Vec::new(),
        }
    }
}
