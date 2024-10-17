//! An extension of `Dataset` that supports search operations.

use distances::Number;
use rayon::iter::ParallelIterator;

use crate::{cluster::ParCluster, dataset::ParDataset, metric::ParMetric, Cluster, Dataset, Metric};

/// A dataset that supports search operations.
pub trait Searchable<I, T: Number, C: Cluster<T>, M: Metric<I, T>>: Dataset<I> {
    /// Returns the distance from a query to the center of the given cluster.
    fn query_to_center(&self, metric: &M, query: &I, cluster: &C) -> T;

    /// Returns the distances from a query to all items in the given cluster.
    fn query_to_all(&self, metric: &M, query: &I, cluster: &C) -> impl Iterator<Item = (usize, T)>;
}

/// A parallel version of [`Searchable`](crate::cakes::dataset::searchable::Searchable).
#[allow(clippy::module_name_repetitions)]
pub trait ParSearchable<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>>:
    ParDataset<I> + Searchable<I, T, C, M>
{
    /// Parallel version of [`Searchable::query_to_center`](crate::cakes::dataset::searchable::Searchable::query_to_center).
    fn par_query_to_center(&self, metric: &M, query: &I, cluster: &C) -> T;

    /// Parallel version of [`Searchable::query_to_all`](crate::cakes::dataset::searchable::Searchable::query_to_all).
    fn par_query_to_all(&self, metric: &M, query: &I, cluster: &C) -> impl ParallelIterator<Item = (usize, T)>;
}
