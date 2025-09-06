//! Ranged nearest neighbor search using a linear scan of the dataset.

use rayon::prelude::*;

use crate::{Cluster, Dataset, DistanceValue, ParCluster, ParDataset};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// Ranged nearest neighbor search using a linear scan of the dataset.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<I, T: DistanceValue, C: Cluster<T>, M: Fn(&I, &I) -> T, D: Dataset<I>> SearchAlgorithm<I, T, C, M, D>
    for RnnLinear<T>
{
    fn name(&self) -> &'static str {
        "RnnLinear"
    }

    fn radius(&self) -> Option<T> {
        Some(self.0)
    }

    fn k(&self) -> Option<usize> {
        None
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.query_to_many(query, root.indices(), metric)
            .into_iter()
            .filter(|&(_, d)| d <= self.0)
            .collect()
    }
}

impl<
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        C: ParCluster<T>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        D: ParDataset<I>,
    > ParSearchAlgorithm<I, T, C, M, D> for RnnLinear<T>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.par_query_to_many(query, root.indices(), metric)
            .into_par_iter()
            .filter(|&(_, d)| d <= self.0)
            .collect()
    }
}
