//! Multiple Sequence Alignment with CLAM

mod aligner;
mod dataset;
mod sequence;

pub use aligner::{ops, Aligner, CostMatrix};
pub use dataset::{Columnar, MSA};
pub use sequence::Sequence;

/// The number of characters.
pub(crate) const NUM_CHARS: usize = 1 + (u8::MAX as usize);
