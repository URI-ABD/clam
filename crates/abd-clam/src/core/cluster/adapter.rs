//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;
use rayon::prelude::*;

use crate::{dataset::ParDataset, Dataset};

use super::{Ball, Cluster, ParCluster};

/// A trait for adapting a `Ball` into another `Cluster`.
#[allow(clippy::module_name_repetitions)]
pub trait BallAdapter<I, U: Number, Din: Dataset<I, U>, Dout: Dataset<I, U>, P: Params<I, U, Din, Dout, Ball<I, U, Din>>>:
    Cluster<I, U, Dout>
{
    /// Adapts this `Cluster` from a `Ball` tree.
    fn from_ball_tree(ball: Ball<I, U, Din>, data: Din, trim: bool) -> (Self, Dout);
}

/// Parallel version of the `BallAdapter` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParBallAdapter<
    I: Send + Sync,
    U: Number,
    Din: ParDataset<I, U>,
    Dout: ParDataset<I, U>,
    P: ParParams<I, U, Din, Dout, Ball<I, U, Din>>,
>: ParCluster<I, U, Dout> + BallAdapter<I, U, Din, Dout, P>
{
    /// Parallel version of the `from_ball_tree` method.
    fn par_from_ball_tree(ball: Ball<I, U, Din>, data: Din, trim: bool) -> (Self, Dout);
}

/// A trait for the parameters to use for adapting a `Ball` into another `Cluster`.
///
/// # Type Parameters:
///
/// - I: The type of instances.
/// - U: The type of distance values.
/// - Din: The type of `Dataset` that the tree was originally built on.
/// - Dout: The type of the `Dataset` that the adapted tree will use.
/// - S: The type of `Cluster` that the tree was originally built on.
pub trait Params<I, U: Number, Din: Dataset<I, U>, Dout: Dataset<I, U>, S: Cluster<I, U, Din>>: Default {
    /// Given the `S` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of `S`.
    #[must_use]
    fn child_params(&self, children: &[S]) -> Vec<Self>;
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
/// - I: The type of instances.
/// - U: The type of distance values.
/// - Din: The type of `Dataset` that the tree was originally built on.
/// - Dout: The type of the `Dataset` that the adapted tree will use.
/// - S: The type of `Cluster` that the tree was originally built on.
/// - P: The type of `Params` to use for adapting the tree.
pub trait Adapter<
    I,
    U: Number,
    Din: Dataset<I, U>,
    Dout: Dataset<I, U>,
    S: Cluster<I, U, Din>,
    P: Params<I, U, Din, Dout, S>,
>: Cluster<I, U, Dout>
{
    /// Creates a new `Cluster` that was adapted from a `S` and a list of children.
    fn new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: P) -> Self;

    /// Performs a task after recursively traversing the tree.
    fn post_traversal(&mut self);

    /// Returns the `Cluster` that was adapted into this `Cluster`. This should not
    /// have any children.
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
    fn adapt_tree(source: S, params: Option<P>) -> Self {
        let params = params.unwrap_or_default();
        let mut cluster = Self::traverse(source, params);
        cluster.post_traversal();
        cluster
    }

    /// Recursively adapts a tree of `S`s into a `Cluster` without any pre- or post-
    /// traversal operations.
    fn traverse(mut source: S, params: P) -> Self {
        let children = source.take_children();

        if children.is_empty() {
            Self::new_adapted(source, Vec::new(), params)
        } else {
            let (arg_extrema, others) = children
                .into_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_iter().map(|(e, c)| (e, *c)).unzip::<_, _, Vec<_>, Vec<_>>();
            let children = params
                .child_params(&children)
                .into_iter()
                .zip(children)
                .map(|(p, c)| Self::adapt_tree(c, Some(p)))
                .zip(arg_extrema)
                .zip(extents)
                .map(|((c, i), d)| (i, d, Box::new(c)))
                .collect();

            Self::new_adapted(source, children, params)
        }
    }

    /// Adapts the tree of `S`s into this `Cluster` in a such a way that we bypass
    /// the recursion limit in Rust.
    fn adapt_tree_iterative(mut source: S, params: Option<P>) -> Self {
        let target_depth = source.depth() + crate::MAX_RECURSION_DEPTH;
        let trimmings = source.trim_at_depth(target_depth);

        let params = params.unwrap_or_default();
        let mut root = Self::traverse(source, params);

        let leaf_params = root
            .leaves()
            .into_iter()
            .filter(|l| l.depth() == target_depth)
            .map(Self::params);

        let trimmings = trimmings
            .into_iter()
            .zip(leaf_params)
            .map(|(children, params)| {
                let (others, children) = children
                    .into_iter()
                    .map(|(i, d, c)| ((i, d), *c))
                    .unzip::<_, _, Vec<_>, Vec<_>>();

                params
                    .child_params(&children)
                    .into_iter()
                    .zip(children)
                    .zip(others)
                    .map(|((p, c), (i, d))| {
                        let c = Self::adapt_tree_iterative(c, Some(p));
                        (i, d, Box::new(c))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        root.graft_at_depth(target_depth, trimmings);
        root.post_traversal();

        root
    }

    /// Recover the source `Cluster` tree that was adapted into this `Cluster`.
    fn recover_source_tree(mut self) -> S {
        let indices = self.source().indices().collect();
        let children = self
            .take_children()
            .into_iter()
            .map(|(i, d, c)| (i, d, Box::new(c.recover_source_tree())))
            .collect();

        let mut source = self.take_source();
        source.set_indices(indices);
        source.set_children(children);
        source
    }
}

/// Parallel version of the `Params` trait.
pub trait ParParams<I: Send + Sync, U: Number, Din: ParDataset<I, U>, Dout: ParDataset<I, U>, S: ParCluster<I, U, Din>>:
    Params<I, U, Din, Dout, S> + Send + Sync
{
    /// Parallel version of the `child_params` method.
    #[must_use]
    fn par_child_params(&self, children: &[S]) -> Vec<Self>;
}

/// Parallel version of the `Adapter` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParAdapter<
    I: Send + Sync,
    U: Number,
    Din: ParDataset<I, U>,
    Dout: ParDataset<I, U>,
    S: ParCluster<I, U, Din>,
    P: ParParams<I, U, Din, Dout, S>,
>: ParCluster<I, U, Dout> + Adapter<I, U, Din, Dout, S, P>
{
    /// Parallel version of the `post_traversal` method.
    fn par_post_traversal(&mut self);

    /// Parallel version of the `adapt` method.
    fn par_adapt_tree(mut source: S, params: Option<P>) -> Self {
        let children = source.take_children();
        let params = params.unwrap_or_default();

        let mut cluster = if children.is_empty() {
            Self::new_adapted(source, Vec::new(), params)
        } else {
            let (arg_extrema, others) = children
                .into_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_iter().map(|(e, c)| (e, *c)).unzip::<_, _, Vec<_>, Vec<_>>();
            let children = params
                .child_params(&children)
                .into_par_iter()
                .zip(children)
                .map(|(p, c)| Self::par_adapt_tree(c, Some(p)))
                .zip(arg_extrema)
                .zip(extents)
                .map(|((c, i), d)| (i, d, Box::new(c)))
                .collect::<Vec<_>>();
            Self::new_adapted(source, children, params)
        };

        cluster.par_post_traversal();

        cluster
    }

    /// Recover the source `Cluster` tree that was adapted into this `Cluster`.
    fn par_recover_source_tree(mut self) -> S {
        let indices = self.source().indices().collect();
        let children = self
            .take_children()
            .into_par_iter()
            .map(|(i, d, c)| (i, d, Box::new(c.par_recover_source_tree())))
            .collect();

        let mut source = self.take_source();
        source.set_indices(indices);
        source.set_children(children);
        source
    }

    /// Adapts the tree of `S`s into this `Cluster` in a such a way that we bypass
    /// the recursion limit in Rust.
    fn par_adapt_tree_iterative(mut source: S, params: Option<P>) -> Self {
        let target_depth = source.depth() + crate::MAX_RECURSION_DEPTH;
        let children = source.trim_at_depth(target_depth);
        let mut root = Self::adapt_tree(source, params);
        let leaf_params = root
            .leaves()
            .into_par_iter()
            .filter(|l| l.depth() == target_depth)
            .map(Self::params)
            .collect::<Vec<_>>();

        let children = children
            .into_par_iter()
            .zip(leaf_params)
            .map(|(children, params)| {
                let (others, children) = children
                    .into_iter()
                    .map(|(i, d, c)| ((i, d), *c))
                    .unzip::<_, _, Vec<_>, Vec<_>>();
                params
                    .child_params(&children)
                    .into_par_iter()
                    .zip(children)
                    .zip(others)
                    .map(|((p, c), (i, d))| {
                        let c = Self::par_adapt_tree_iterative(c, Some(p));
                        (i, d, Box::new(c))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        root.graft_at_depth(target_depth, children);
        root
    }
}
