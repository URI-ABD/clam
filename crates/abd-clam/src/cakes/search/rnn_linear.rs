//! Ranged nearest neighbor search using a linear scan of the dataset.

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric,
};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// Ranged nearest neighbor search using a linear scan of the dataset.
pub struct RnnLinear<T: Number>(pub T);

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for RnnLinear<T>
{
    fn name(&self) -> &str {
        "RnnLinear"
    }

    fn radius(&self) -> Option<T> {
        Some(self.0)
    }

    fn k(&self) -> Option<usize> {
        None
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.query_to_all(metric, query, root)
            .filter(|&(_, d)| d <= self.0)
            .collect()
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for RnnLinear<T>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.par_query_to_all(metric, query, root)
            .filter(|&(_, d)| d <= self.0)
            .collect()
    }
}
