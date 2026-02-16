//! A single connected component in a CHAODA graph.

use rayon::prelude::*;

use crate::DistanceValue;

use super::{Component, Node};

impl<T: DistanceValue + Send + Sync> Component<T> {
    /// Returns a parallel iterator over the nodes in this component.
    #[must_use]
    pub fn par_iter_nodes(&self) -> impl ParallelIterator<Item = &Node<T>> {
        self.nodes.par_iter().map(|(_, v)| v)
    }
}
