//! k-NN search using a linear scan of the dataset.

use distances::Number;
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric, SizedHeap,
};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// k-NN search using a linear scan of the dataset.
pub struct KnnLinear(pub usize);

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for KnnLinear
{
    fn name(&self) -> &str {
        "KnnLinear"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.query_to_all(metric, query, root)
            .fold(SizedHeap::new(Some(self.0)), |mut hits, (i, d)| {
                hits.push((d, i));
                hits
            })
            .items()
            .map(|(d, i)| (i, d))
            .collect()
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for KnnLinear
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.par_query_to_all(metric, query, root)
            .fold(
                || SizedHeap::new(Some(self.0)),
                |mut hits, (i, d)| {
                    hits.push((d, i));
                    hits
                },
            )
            .reduce(
                || SizedHeap::new(Some(self.0)),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            )
            .par_items()
            .map(|(d, i)| (i, d))
            .collect()
    }
}
