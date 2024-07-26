//! Traits to adapt a `Ball` into other `Cluster` types.

use distances::Number;

use super::{Ball, Children, Cluster};

/// A trait for adapting a `Ball` into another `Cluster` type.
pub trait ClusterAdaptor<U: Number>: Cluster<U> {
    /// Adapts a `Ball` into a `Cluster`, using arbitrary parameters.
    fn adapt<P>(ball: &Ball<U>, params: P) -> Self;

    /// Returns the `Ball` that was adapted into this `Cluster`. This should not
    /// have any children.
    fn ball(&self) -> &Ball<U>;

    /// Returns the `Ball` mutably that was adapted into this `Cluster`. This
    /// should not have any children.
    fn ball_mut(&mut self) -> &mut Ball<U>;

    /// Returns the `Children` of the `Cluster`.
    #[must_use]
    fn children(&self) -> Option<&Children<U, Self>>
    where
        Self: Sized;

    /// Returns the children of the `Cluster` as mutable references.
    #[must_use]
    fn children_mut(&mut self) -> Option<&mut Children<U, Self>>
    where
        Self: Sized;
}
