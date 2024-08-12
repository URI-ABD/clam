//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use crate::{dataset::ParDataset, Dataset};

use super::{Ball, Cluster, ParCluster};

/// A trait for adapting a `Ball` into another `Cluster`.
#[allow(clippy::module_name_repetitions)]
pub trait BallAdapter<I, U: Number, Din: Dataset<I, U>, Dout: Dataset<I, U>, P: Params<I, U, Din, Dout, Ball<I, U, Din>>>:
    Cluster<I, U, Dout>
{
    /// Adapts this `Cluster` from a `Ball` tree.
    fn from_ball_tree(ball: Ball<I, U, Din>, data: Din) -> (Self, Dout);
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
    fn par_from_ball_tree(ball: Ball<I, U, Din>, data: Din) -> (Self, Dout);
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
    fn child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self>;
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
    /// Recursively adapts a tree of `S`s into a `Cluster`.
    ///
    /// # Arguments
    ///
    /// - `source`: The `S` to adapt.
    /// - `params`: The parameters to use for adapting `S`. If `None`, assume
    ///   that `S` is a root `Cluster` and use the default parameters.
    ///
    /// # Returns
    ///
    /// - The adapted `Cluster`.
    /// - A list of indices of `S`.
    fn adapt_tree(source: S, params: Option<P>) -> (Self, Vec<usize>);

    /// Creates a new `Cluster` that was adapted from a `S` and a list of children.
    fn new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: P) -> Self;

    /// Returns the `Cluster` that was adapted into this `Cluster`. This should not
    /// have any children.
    fn source(&self) -> &S;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn source_mut(&mut self) -> &mut S;
}

/// Parallel version of the `Params` trait.
pub trait ParParams<I: Send + Sync, U: Number, Din: ParDataset<I, U>, Dout: ParDataset<I, U>, S: ParCluster<I, U, Din>>:
    Params<I, U, Din, Dout, S> + Send + Sync
{
    /// Parallel version of the `child_params` method.
    #[must_use]
    fn par_child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self>;
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
    /// Parallel version of the `adapt` method.
    fn par_adapt_tree(source: S, params: Option<P>) -> (Self, Vec<usize>);
}
