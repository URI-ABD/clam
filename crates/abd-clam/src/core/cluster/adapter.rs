//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;
use rayon::prelude::*;

use crate::{dataset::ParDataset, metric::ParMetric, utils, Dataset, Metric};

use super::{Ball, Cluster, ParCluster};

/// Used for adapting a `Ball` into another `Cluster`.
///
/// # Type Parameters:
///
/// - I: The items.
/// - T: The distance values.
/// - D: The `Dataset` that the tree was originally built on.
/// - S: The `Cluster` that the tree was originally built on.
///
/// # Examples
///
/// See:
///
/// - [`Offset`](crate::cakes::Offset)
/// - [`SquishCosts`](crate::pancakes::SquishCosts)
pub trait Params<I, T: Number, D: Dataset<I>, S: Cluster<T>>: Default {
    /// Given the `S` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of `S`.
    #[must_use]
    fn child_params<M: Metric<I, T>>(&self, children: &[S], data: &D, metric: &M) -> Vec<Self>;
}

/// Parallel version of [`Params`](crate::core::cluster::adapter::Params).
///
/// # Examples
///
/// See:
///
/// - [`Offset`](crate::cakes::Offset)
/// - [`SquishCosts`](crate::pancakes::SquishCosts)
pub trait ParParams<I: Send + Sync, T: Number, D: ParDataset<I>, S: ParCluster<T>>:
    Params<I, T, D, S> + Send + Sync
{
    /// Parallel version of [`Params::child_params`](crate::core::cluster::adapter::Params::child_params).
    #[must_use]
    fn par_child_params<M: ParMetric<I, T>>(&self, children: &[S], data: &D, metric: &M) -> Vec<Self>;
}

/// A trait for adapting a `Ball` into another `Cluster`.
///
/// # Examples
///
/// See:
///
/// - [`PermutedBall`](crate::cakes::PermutedBall)
/// - [`SquishyBall`](crate::pancakes::SquishyBall)
#[allow(clippy::module_name_repetitions)]
pub trait BallAdapter<I, T: Number, Din: Dataset<I>, Dout: Dataset<I>, P: Params<I, T, Din, Ball<T>>>:
    Cluster<T> + Sized
{
    /// Adapts this `Cluster` from a `Ball` tree.
    fn from_ball_tree<M: Metric<I, T>>(ball: Ball<T>, data: Din, metric: &M) -> (Self, Dout);
}

/// Parallel version of the [`BallAdapter`](crate::core::cluster::adapter::BallAdapter)
/// trait.
///
/// # Examples
///
/// See:
///
/// - [`PermutedBall`](crate::cakes::PermutedBall)
/// - [`SquishyBall`](crate::pancakes::SquishyBall)
#[allow(clippy::module_name_repetitions)]
pub trait ParBallAdapter<
    I: Send + Sync,
    T: Number,
    Din: ParDataset<I>,
    Dout: ParDataset<I>,
    P: ParParams<I, T, Din, Ball<T>>,
>: ParCluster<T> + BallAdapter<I, T, Din, Dout, P>
{
    /// Parallel version of [`BallAdapter::from_ball_tree`](crate::core::cluster::adapter::BallAdapter::from_ball_tree).
    fn par_from_ball_tree<M: ParMetric<I, T>>(ball: Ball<T>, data: Din, metric: &M) -> (Self, Dout);
}

/// A trait for adapting one `Cluster` type into another `Cluster` type.
///
/// The workflow for adapting a `Cluster` is as follows:
///
/// 1. If `S` implements `Partition`, build a tree of `S`s. Otherwise, adapt `S`
///    from another `Cluster` type.
/// 2. Adapt the tree of `S`s into this `Cluster` type.
///
/// # Type Parameters:
///
/// - I: The items.
/// - T: The distance values.
/// - Din: The `Dataset` that the tree was originally built on.
/// - Dout: The the `Dataset` that the adapted tree will use.
/// - S: The `Cluster` that the tree was originally built on.
/// - P: The `Params` to use for adapting the tree.
///
/// # Examples
///
/// See:
///
/// - [`PermutedBall`](crate::cakes::PermutedBall)
/// - [`SquishyBall`](crate::pancakes::SquishyBall)
pub trait Adapter<I, T: Number, Din: Dataset<I>, Dout: Dataset<I>, S: Cluster<T>, P: Params<I, T, Din, S>>:
    Cluster<T> + Sized
{
    /// Creates a new `Cluster` that was adapted from a `S` and a list of
    /// children.
    fn new_adapted<M: Metric<I, T>>(source: S, children: Vec<Box<Self>>, params: P, data: &Din, metric: &M) -> Self;

    /// Performs a task after recursively traversing the tree.
    fn post_traversal(&mut self);

    /// Returns the `Cluster` that was adapted into this `Cluster`. This should
    /// not have any children.
    fn source(&self) -> &S;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn source_mut(&mut self) -> &mut S;

    /// Provides ownership of the underlying source `Cluster`.
    fn take_source(self) -> S;

    /// Returns the params used to adapt the `Cluster`
    fn params(&self) -> &P;

    /// Recursively adapts a tree of `S`s into a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `S` to adapt.
    /// - `params`: The parameters to use for adapting `S`. If `None`, assume
    ///   that `S` is a root `Cluster` and use the default parameters.
    /// - `data`: The `Dataset` that the tree was built on.
    /// - `metric`: The `Metric` to use for distance calculations.
    fn adapt_tree<M: Metric<I, T>>(source: S, params: Option<P>, data: &Din, metric: &M) -> Self {
        let params = params.unwrap_or_default();
        let mut cluster = Self::traverse(source, params, data, metric);
        cluster.post_traversal();
        cluster
    }

    /// Recursively adapts a tree of `S`s into a `Cluster` without any pre- or
    /// post- traversal operations.
    fn traverse<M: Metric<I, T>>(mut source: S, params: P, data: &Din, metric: &M) -> Self {
        let children = source.take_children().into_iter().map(|c| *c).collect::<Vec<_>>();

        if children.is_empty() {
            Self::new_adapted(source, Vec::new(), params, data, metric)
        } else {
            let children = params
                .child_params(&children, data, metric)
                .into_iter()
                .zip(children)
                .map(|(p, c)| Self::adapt_tree(c, Some(p), data, metric))
                .map(Box::new)
                .collect();

            Self::new_adapted(source, children, params, data, metric)
        }
    }

    /// Adapts the tree of `S`s into this `Cluster` in a such a way that we
    /// bypass the recursion limit in Rust.
    fn adapt_tree_iterative<M: Metric<I, T>>(mut source: S, params: Option<P>, data: &Din, metric: &M) -> Self {
        let target_depth = source.depth() + utils::max_recursion_depth();
        let trimmings = source.trim_at_depth(target_depth);

        let mut root = Self::adapt_tree(source, params, data, metric);

        let leaf_params = root
            .leaves()
            .into_iter()
            .filter(|l| l.depth() == target_depth)
            .map(Self::params)
            .collect::<Vec<_>>();

        let trimmings = trimmings
            .into_iter()
            .zip(leaf_params)
            .map(|(children, params)| {
                let children = children.into_iter().map(|c| *c).collect::<Vec<_>>();
                params
                    .child_params(&children, data, metric)
                    .into_iter()
                    .zip(children)
                    .map(|(p, c)| Self::adapt_tree_iterative(c, Some(p), data, metric))
                    .map(Box::new)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        root.graft_at_depth(target_depth, trimmings);

        root
    }

    /// Recover the source `Cluster` tree that was adapted into this `Cluster`.
    fn recover_source_tree(mut self) -> S {
        let indices = self.source().indices();
        let children = self
            .take_children()
            .into_iter()
            .map(|c| c.recover_source_tree())
            .map(Box::new)
            .collect();

        let mut source = self.take_source();
        source.set_indices(&indices);
        source.set_children(children);
        source
    }
}

/// Parallel version of [`Adapter`](crate::core::cluster::adapter::Adapter).
///
/// # Examples
///
/// See:
///
/// - [`PermutedBall`](crate::cakes::PermutedBall)
/// - [`SquishyBall`](crate::pancakes::SquishyBall)
#[allow(clippy::module_name_repetitions)]
pub trait ParAdapter<
    I: Send + Sync,
    T: Number,
    Din: ParDataset<I>,
    Dout: ParDataset<I>,
    S: ParCluster<T>,
    P: ParParams<I, T, Din, S>,
>: ParCluster<T> + Adapter<I, T, Din, Dout, S, P>
{
    /// Parallel version of [`Adapter::new_adapted`](crate::core::cluster::adapter::Adapter::new_adapted).
    fn par_new_adapted<M: ParMetric<I, T>>(
        source: S,
        children: Vec<Box<Self>>,
        params: P,
        data: &Din,
        metric: &M,
    ) -> Self;

    /// Parallel version of [`Adapter::adapt_tree`](crate::core::cluster::adapter::Adapter::adapt_tree).
    fn par_adapt_tree<M: ParMetric<I, T>>(mut source: S, params: Option<P>, data: &Din, metric: &M) -> Self {
        let children = source.take_children().into_iter().map(|c| *c).collect::<Vec<_>>();
        let params = params.unwrap_or_default();

        let mut cluster = if children.is_empty() {
            Self::par_new_adapted(source, Vec::new(), params, data, metric)
        } else {
            let children = params
                .child_params(&children, data, metric)
                .into_par_iter()
                .zip(children)
                .map(|(p, c)| Self::par_adapt_tree(c, Some(p), data, metric))
                .map(Box::new)
                .collect();
            Self::par_new_adapted(source, children, params, data, metric)
        };

        cluster.post_traversal();

        cluster
    }

    /// Parallel version of [`Adapter::recover_source_tree`](crate::core::cluster::adapter::Adapter::recover_source_tree).
    fn par_recover_source_tree(mut self) -> S {
        let indices = self.source().indices();
        let children = self
            .take_children()
            .into_par_iter()
            .map(|c| c.par_recover_source_tree())
            .map(Box::new)
            .collect();

        let mut source = self.take_source();
        source.set_indices(&indices);
        source.set_children(children);
        source
    }

    /// Parallel version of [`Adapter::adapt_tree_iterative`](crate::core::cluster::adapter::Adapter::adapt_tree_iterative).
    fn par_adapt_tree_iterative<M: ParMetric<I, T>>(mut source: S, params: Option<P>, data: &Din, metric: &M) -> Self {
        let target_depth = source.depth() + utils::max_recursion_depth();
        let trimmings = source.trim_at_depth(target_depth);

        let mut root = Self::par_adapt_tree(source, params, data, metric);

        let leaf_params = root
            .leaves()
            .into_par_iter()
            .filter(|l| l.depth() == target_depth)
            .map(Self::params)
            .collect::<Vec<_>>();

        let trimmings = trimmings
            .into_par_iter()
            .zip(leaf_params)
            .map(|(children, params)| {
                let children = children.into_iter().map(|c| *c).collect::<Vec<_>>();
                params
                    .child_params(&children, data, metric)
                    .into_par_iter()
                    .zip(children)
                    .map(|(p, c)| Self::par_adapt_tree_iterative(c, Some(p), data, metric))
                    .map(Box::new)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        root.graft_at_depth(target_depth, trimmings);

        root
    }
}
