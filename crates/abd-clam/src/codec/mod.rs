//! Compression and Decompression

mod criteria;
mod dataset;
mod squishy_ball;

use distances::number::UInt;

pub use dataset::{decode_general, encode_general, GenomicDataset, SquishyDataset};
pub use squishy_ball::SquishyBall;

use crate::{Instance, Tree};

/// A function that encodes a `Instance` into a `Box<[u8]>`.
pub type EncoderFn<I> = fn(&I, &I) -> Result<Box<[u8]>, String>;

/// A function that decodes a `Instance` from a `&[u8]`.
pub type DecoderFn<I> = fn(&I, &[u8]) -> Result<I, String>;

impl<I: Instance, U: UInt, D: SquishyDataset<I, U>> Tree<I, U, D, SquishyBall<U>> {
    /// Recursively estimates and sets the costs of recursive and unitary compression in the subtree.
    #[must_use]
    pub fn estimate_costs(mut self, data: &D) -> Self {
        self.root.calculate_costs(data);
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
