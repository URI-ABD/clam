//! Multiple Sequence Alignment with CLAM

mod aligner;
mod dataset;

pub use aligner::{ops, Aligner, CostMatrix};
pub use dataset::{Columns, MSA};

/// The number of characters.
pub(crate) const NUM_CHARS: usize = 1 + (u8::MAX as usize);
