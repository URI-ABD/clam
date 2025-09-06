//! k-NN search using a linear scan of the dataset.

use rayon::prelude::*;

use crate::{Cluster, Dataset, DistanceValue, ParCluster, ParDataset, SizedHeap};

use super::{ParSearchAlgorithm, SearchAlgorithm};

/// k-NN search using a linear scan of the dataset.
pub struct KnnLinear(pub usize);

impl<I, T: DistanceValue, C: Cluster<T>, M: Fn(&I, &I) -> T, D: Dataset<I>> SearchAlgorithm<I, T, C, M, D>
    for KnnLinear
{
    fn name(&self) -> &'static str {
        "KnnLinear"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.query_to_many(query, root.indices(), metric)
            .into_iter()
            .fold(SizedHeap::new(Some(self.0)), |mut hits, (i, d)| {
                hits.push((d, i));
                hits
            })
            .items()
            .map(|(d, i)| (i, d))
            .collect()
    }
}

impl<
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        C: ParCluster<T>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        D: ParDataset<I>,
    > ParSearchAlgorithm<I, T, C, M, D> for KnnLinear
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        data.par_query_to_many(query, root.indices(), metric)
            .into_par_iter()
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
