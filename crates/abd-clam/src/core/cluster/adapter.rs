//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use crate::{dataset::ParDataset, Dataset};

use super::{Cluster, ParCluster};

/// A trait for the parameters to use for adapting a `Ball` into another `Cluster`.
///
/// # Type Parameters:
///
/// - I: The type of instances.
/// - U: The type of distance values.
/// - Din: The type of `Dataset` that the tree was originally built on.
/// - Dout: The type of the `Dataset` that the adapted tree will use.
pub trait Params<I, U: Number, Din: Dataset<I, U>, Dout: Dataset<I, U>, S: Cluster<I, U, Din>>: Default {
    /// Given the `S` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of `S`.
    #[must_use]
    fn child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self>;
}

/// A trait for adapting one `Cluster` type into another `Cluster` type.
pub trait Adapter<
    I,
    U: Number,
    Din: Dataset<I, U>,
    Dout: Dataset<I, U>,
    S: Cluster<I, U, Din>,
    P: Params<I, U, Din, Dout, S>,
>: Cluster<I, U, Dout>
{
    /// Adapts a tree of `S`s into a `Cluster`.
    fn adapt(source: S, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;

    /// Creates a new `Cluster` that was adapted from a `S` and a list of children.
    fn newly_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: P) -> Self;

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
>: Adapter<I, U, Din, Dout, S, P> + Send + Sync
{
    /// Parallel version of the `adapt` method.
    fn par_adapt(source: S, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;
}
