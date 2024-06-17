//! Compression and Decompression

mod criteria;
mod dataset;
mod squishy_ball;

pub use criteria::{CompressionCriteria, CompressionCriterion};
pub use dataset::{decode_general, encode_general, GenomicDataset, SquishyDataset};
pub use squishy_ball::SquishyBall;

/// A function that encodes a `Instance` into a `Box<[u8]>`.
pub type EncoderFn<I> = fn(&I, &I) -> Result<Box<[u8]>, String>;

/// A function that decodes a `Instance` from a `&[u8]`.
pub type DecoderFn<I> = fn(&I, &[u8]) -> Result<I, String>;
