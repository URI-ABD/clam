use core::{cmp::Ordering, f64::EPSILON};

use distances::Number;

use super::RnnAlgorithm;
use crate::{cluster::Tree, dataset::Dataset, utils::helpers};

#[derive(Clone, Copy, Debug)]
pub enum KnnAlgorithm {
    Linear,
    RepeatedRnn,
}

impl Default for KnnAlgorithm {
    fn default() -> Self {
        Self::RepeatedRnn
    }
}

impl KnnAlgorithm {
    pub fn search<T, U, D>(&self, query: T, k: usize, tree: &Tree<T, U, D>) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        match self {
            Self::Linear => Self::linear_search(tree.data(), query, k, tree.indices()),
            Self::RepeatedRnn => Self::knn_by_rnn(tree, query, k),
        }
    }

    pub(crate) fn linear_search<T, U, D>(data: &D, query: T, k: usize, indices: &[usize]) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let distances = data.query_to_many(query, indices);
        let mut hits = indices.iter().copied().zip(distances.into_iter()).collect::<Vec<_>>();
        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Less));
        hits[..k].to_vec()
    }

    pub(crate) fn knn_by_rnn<T, U, D>(tree: &Tree<T, U, D>, query: T, k: usize) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let mut radius = EPSILON + tree.radius().as_f64() / tree.cardinality().as_f64();
        let mut hits = RnnAlgorithm::clustered_search(tree, query, U::from(radius));
        const MULTIPLIER: f64 = 2.0;

        while hits.is_empty() {
            radius *= MULTIPLIER;
            hits = RnnAlgorithm::clustered_search(tree, query, U::from(radius));
        }

        while hits.len() < k {
            let distances = hits.iter().map(|(_, d)| *d).collect::<Vec<_>>();
            let lfd = helpers::compute_lfd(U::from(radius), &distances);
            let factor = (k.as_f64() / hits.len().as_f64()).powf(1. / (lfd + EPSILON));
            assert!(factor > 1.);
            radius *= if factor < MULTIPLIER { factor } else { MULTIPLIER };
            hits = RnnAlgorithm::clustered_search(tree, query, U::from(radius));
        }

        hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        hits[..k].to_vec()
    }
}
