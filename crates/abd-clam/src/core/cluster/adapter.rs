//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use super::{Ball, Children, Cluster};

/// A trait for the parameters to use for adapting a `Ball` into another `Cluster`.
pub trait Params<U: Number>: Default + Copy {
    /// Given the `Ball` that was adapted into a `Cluster`, returns parameters
    /// to use for adapting the children of the `Ball`.
    #[must_use]
    fn child_params(&self, c: &Ball<U>) -> Self;
}

/// A trait for adapting a `Ball` into another `Cluster` type.
///
/// # Parameters
///
/// - `U`: The type of the distance values.
/// - `P`: The type of the parameters to use for the adaptation.
pub trait Adapter<U: Number, P: Params<U>>: Cluster<U> {
    /// Adapts a tree of `Ball`s into a `Cluster`.
    fn adapt(root: Ball<U>) -> Self
    where
        Self: Sized,
    {
        let (ball, children) = root.take_children();
        let params = P::default();

        if let Some(children) = children {
            let params = params.child_params(&ball);
            let (balls, arg_poles, polar_distances) = children.take();
            let children = balls.into_iter().map(|ball| Self::adapt_one(ball, params)).collect();
            let children = Children::new(children, arg_poles, polar_distances);
            Self::adapt_one(ball, params).set_children(children)
        } else {
            Self::adapt_one(ball, params)
        }
    }

    /// Adapts a `Ball` into a `Cluster`. This should not have any children.
    fn adapt_one(ball: Ball<U>, params: P) -> Self
    where
        Self: Sized;

    /// Returns the `Ball` that was adapted into this `Cluster`. This should not
    /// have any children.
    fn ball(&self) -> &Ball<U>;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn ball_mut(&mut self) -> &mut Ball<U>;
}
