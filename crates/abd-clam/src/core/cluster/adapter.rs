//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use super::Cluster;

/// A trait for the parameters to use for adapting a `Ball` into another `Cluster`.
pub trait Params<U: Number, S: Cluster<U>>: Default {
    /// Given the `Ball` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of the `Ball`.
    #[must_use]
    fn child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self>;
}

/// A trait for adapting one `Cluster` type into another `Cluster` type.
///
/// # Parameters
///
/// - `U`: The type of the distance values.
/// - `S`: The type of the `Cluster` to adapt from.
/// - `P`: The type of the parameters to use for the adaptation.
pub trait Adapter<U: Number, S: Cluster<U>, P: Params<U, S>>: Cluster<U> {
    /// Adapts a tree of `Ball`s into a `Cluster`.
    fn adapt(source: S, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;

    /// Returns the `Cluster` that was adapted into this `Cluster`. This should not
    /// have any children.
    fn inner(&self) -> &S;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn inner_mut(&mut self) -> &mut S;
}

/// Parallel version of the `Params` trait.
pub trait ParParams<U: Number, S: Cluster<U>>: Params<U, S> + Send + Sync {
    /// Parallel version of the `child_params` method.
    #[must_use]
    fn par_child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self>;
}

/// Parallel version of the `Adapter` trait.
#[allow(clippy::module_name_repetitions)]
pub trait ParAdapter<U: Number, S: Cluster<U>, P: ParParams<U, S>>: Adapter<U, S, P> + Send + Sync {
    /// Parallel version of the `adapt` method.
    fn par_adapt(source: S, params: Option<P>) -> (Self, Vec<usize>)
    where
        Self: Sized;
}
