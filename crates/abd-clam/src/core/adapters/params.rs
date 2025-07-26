//! The parameters used for adapting one `Cluster` into another `Cluster`.

use distances::Number;

use crate::{cluster::ParCluster, dataset::ParDataset, metric::ParMetric, Cluster, Dataset, Metric};

/// Used for adapting a `Cluster` into another `Cluster`.
///
/// # Type Parameters:
///
/// - I: The items.
/// - T: The distance values.
/// - D: The `Dataset` that the tree was originally built on.
/// - S: The `Cluster` that the tree was originally built on.
/// - M: The `Metric` that the tree was originally built with.
pub trait Params<I, T: Number, D: Dataset<I>, S: Cluster<T>, M: Metric<I, T>>: Default {
    /// Given the `S` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of `S`.
    #[must_use]
    fn child_params(&self, children: &[S], data: &D, metric: &M) -> Vec<Self>;
}

/// Parallel version of [`Params`](Params).
pub trait ParParams<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>, M: ParMetric<I, T>>:
    Params<I, T, D, S, M> + Send + Sync
{
    /// Parallel version of [`Params::child_params`](Params::child_params).
    #[must_use]
    fn par_child_params(&self, children: &[S], data: &D, metric: &M) -> Vec<Self>;
}
