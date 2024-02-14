//! Compression and Decompression

mod dataset;
mod squishy_ball;

use distances::number::Int;

pub use dataset::SquishyDataset;
pub use squishy_ball::SquishyBall;

use crate::{Instance, Tree};

impl<I: Instance, U: Int, D: SquishyDataset<I, U>> Tree<I, U, D, SquishyBall<U>> {
    /// Recursively estimates and sets the costs of recursive and unitary compression in the subtree.
    #[must_use]
    pub fn estimate_costs(mut self, data: &D) -> Self {
        self.root.estimate_costs(data);
        self
    }

    /// Compresses the `SquishedBall` recursively by descending into the subtree.
    ///
    /// The centers of the children are encoded in terms of the center of the `SquishedBall`.
    #[must_use]
    pub fn compress_recursive(&self) -> Self {
        todo!()
    }

    /// Compresses the `SquishedBall` without descending into the subtree.
    ///
    /// Every `Instance` is encoded in terms of the center of the `SquishedBall`.
    #[must_use]
    pub fn compress_unitary(&self) -> Self {
        // TODO: Implement encode/decode for metrics, probably as an extension trait for `Dataset`.
        todo!()
    }
}
