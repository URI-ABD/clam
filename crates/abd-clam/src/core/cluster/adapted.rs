//! Traits for `Cluster`s that have been adapted from other `Cluster`s.

use crate::DistanceValue;

use super::Cluster;

/// A `Cluster` that has been adapted from a different `Cluster`.
pub trait Adapted<T: DistanceValue, S: Cluster<T>>: Cluster<T> {
    /// Returns the `Cluster` that was adapted into this `Cluster`.
    fn source(&self) -> &S;

    /// Returns the `Cluster` that was adapted into this `Cluster`.
    fn source_mut(&mut self) -> &mut S;

    /// Provides ownership of the underlying source `Cluster`.
    fn take_source(self) -> S;
}
