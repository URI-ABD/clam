//! K-Nearest Neighbor search algorithms.
//!
//! The stable algorithms are `Linear` and `RepeatedRnn`, with the default being
//! `RepeatedRnn`. We will experiment with other algorithms in the future, and they
//! will be added to this module as they are being implemented. They should not be
//! considered stable until they are documented as such.

use core::{cmp::Ordering, f64::EPSILON};

use distances::Number;

use crate::{utils, Dataset, RnnAlgorithm, Tree};

/// The multiplier to use for increasing the radius in the repeated RNN algorithm.
const MULTIPLIER: f64 = 2.0;

/// The algorithm to use for K-Nearest Neighbor search.
///
/// The default is `RepeatedRnn`, as determined by the benchmarks in the crate.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub enum KnnAlgorithm {
    /// Use linear search on the entire dataset. This is a stable algorithm.
    Linear,
    /// Use a repeated RNN search, increasing the radius until enough neighbors
    /// are found. This is a stable algorithm.
    ///
    /// Search starts with a radius equal to the radius of the tree divided by
    /// the cardinality of the dataset. If no neighbors are found, the radius is
    /// increased by a factor of 2 until at least one neighbor is found. Then,
    /// the radius is increased by a factor determined by the local fractal
    /// dimension of the neighbors found until enough neighbors are found. This
    /// factor is capped at 2. Once enough neighbors are found, the neighbors
    /// are sorted by distance and the first `k` neighbors are returned. Ties
    /// are broken arbitrarily.
    RepeatedRnn,
}

impl Default for KnnAlgorithm {
    fn default() -> Self {
        Self::RepeatedRnn
    }
}

impl KnnAlgorithm {
    /// Searches for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `tree` - The tree to search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn search<T, U, D>(self, query: &T, k: usize, tree: &Tree<T, U, D>) -> Vec<(usize, U)>
    where
        T: Send + Sync,
        U: Number,
        D: Dataset<T, U>,
    {
        match self {
            Self::Linear => Self::linear_search(tree.data(), query, k, tree.indices()),
            Self::RepeatedRnn => Self::knn_by_rnn(tree, query, k),
        }
    }

    /// Linear search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `indices` - The indices to search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn linear_search<T, U, D>(data: &D, query: &T, k: usize, indices: &[usize]) -> Vec<(usize, U)>
    where
        T: Send + Sync,
        U: Number,
        D: Dataset<T, U>,
    {
        let distances = data.query_to_many(query, indices);
        let mut hits = indices.iter().copied().zip(distances.into_iter()).collect::<Vec<_>>();
        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less));
        hits[..k].to_vec()
    }

    /// K-Nearest Neighbor search using a repeated RNN search.
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree to search.
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn knn_by_rnn<T, U, D>(tree: &Tree<T, U, D>, query: &T, k: usize) -> Vec<(usize, U)>
    where
        T: Send + Sync,
        U: Number,
        D: Dataset<T, U>,
    {
        let mut radius = EPSILON + tree.radius().as_f64() / tree.cardinality().as_f64();
        let [mut confirmed, mut straddlers] =
            RnnAlgorithm::tree_search(tree.data(), tree.root(), query, U::from(radius));

        let mut num_hits = confirmed
            .iter()
            .chain(straddlers.iter())
            .map(|&(c, _)| c.cardinality)
            .sum::<usize>();

        while num_hits == 0 {
            radius *= MULTIPLIER;
            [confirmed, straddlers] = RnnAlgorithm::tree_search(tree.data(), tree.root(), query, U::from(radius));
            num_hits = confirmed
                .iter()
                .chain(straddlers.iter())
                .map(|&(c, _)| c.cardinality)
                .sum::<usize>();
        }

        while num_hits < k {
            let lfd = utils::mean(
                &confirmed
                    .iter()
                    .chain(straddlers.iter())
                    .map(|&(c, _)| c.lfd)
                    .collect::<Vec<_>>(),
            );
            let factor = (k.as_f64() / num_hits.as_f64()).powf(1. / (lfd + EPSILON));
            assert!(factor > 1.);
            radius *= if factor < MULTIPLIER { factor } else { MULTIPLIER };
            [confirmed, straddlers] = RnnAlgorithm::tree_search(tree.data(), tree.root(), query, U::from(radius));
            num_hits = confirmed
                .iter()
                .chain(straddlers.iter())
                .map(|&(c, _)| c.cardinality)
                .sum::<usize>();
        }

        let mut hits = confirmed
            .into_iter()
            .chain(straddlers.into_iter())
            .flat_map(|(c, d)| {
                let indices = c.indices(tree.data());
                let distances = if c.is_singleton() {
                    vec![d; c.cardinality]
                } else {
                    tree.data().query_to_many(query, indices)
                };
                indices.iter().copied().zip(distances.into_iter())
            })
            .collect::<Vec<_>>();

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater));
        hits[..k].to_vec()
    }
}
