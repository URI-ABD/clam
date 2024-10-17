//! Dataset extensions for search.

use std::collections::HashMap;

use distances::Number;
use rayon::prelude::*;

use crate::{cluster::ParCluster, metric::ParMetric, Cluster, Dataset, FlatVec, Metric};

mod hinted;
mod searchable;

#[allow(clippy::module_name_repetitions)]
pub use hinted::{HintedDataset, ParHintedDataset};
pub use searchable::{ParSearchable, Searchable};

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, Me> Searchable<I, T, C, M> for FlatVec<I, Me> {
    fn query_to_center(&self, metric: &M, query: &I, cluster: &C) -> T {
        metric.distance(query, self.get(cluster.arg_center()))
    }

    #[inline(never)]
    fn query_to_all(&self, metric: &M, query: &I, cluster: &C) -> impl Iterator<Item = (usize, T)> {
        cluster
            .indices()
            .into_iter()
            .map(|i| (i, metric.distance(query, self.get(i))))
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, Me: Send + Sync> ParSearchable<I, T, C, M>
    for FlatVec<I, Me>
{
    fn par_query_to_center(&self, metric: &M, query: &I, cluster: &C) -> T {
        metric.par_distance(query, self.get(cluster.arg_center()))
    }

    fn par_query_to_all(
        &self,
        metric: &M,
        query: &I,
        cluster: &C,
    ) -> impl rayon::prelude::ParallelIterator<Item = (usize, T)> {
        cluster
            .par_indices()
            .map(|i| (i, metric.par_distance(query, self.get(i))))
    }
}

#[allow(clippy::implicit_hasher)]
impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, Me> HintedDataset<I, T, C, M>
    for FlatVec<I, (Me, HashMap<usize, T>)>
{
    fn hints_for(&self, i: usize) -> &HashMap<usize, T> {
        &self.metadata[i].1
    }

    fn hints_for_mut(&mut self, i: usize) -> &mut HashMap<usize, T> {
        &mut self.metadata[i].1
    }
}

#[allow(clippy::implicit_hasher)]
impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, Me: Send + Sync> ParHintedDataset<I, T, C, M>
    for FlatVec<I, (Me, HashMap<usize, T>)>
{
}
