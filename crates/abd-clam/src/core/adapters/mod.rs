//! `Cluster`s can be adapted to other types of `Cluster`s. This module contains
//! some traits that allow for this.

use distances::Number;

use super::Cluster;

mod adapt;
mod ball_adapter;
mod params;

pub use adapt::{Adapter, ParAdapter};
pub use ball_adapter::{BallAdapter, ParBallAdapter};
pub use params::{ParParams, Params};

/// A `Cluster` that has been adapted from a different `Cluster`.
pub trait Adapted<T: Number, S: Cluster<T>>: Cluster<T> {
    /// Returns the `Cluster` that was adapted into this `Cluster`.
    fn source(&self) -> &S;

    /// Returns the `Cluster` that was adapted into this `Cluster`.
    fn source_mut(&mut self) -> &mut S;

    /// Provides ownership of the underlying source `Cluster`.
    fn take_source(self) -> S;
}
