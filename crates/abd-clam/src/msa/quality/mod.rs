//! Quality and accuracy metrics for MSAs.

pub mod col_major;
pub mod row_major;

/// The square root threshold for sub-sampling.
pub(crate) const SQRT_THRESH: usize = 1000;
/// The logarithmic threshold for sub-sampling.
pub(crate) const LOG2_THRESH: usize = 100_000;
