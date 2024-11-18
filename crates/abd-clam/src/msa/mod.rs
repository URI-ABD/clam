//! Multiple Sequence Alignment with CLAM

use core::ops::{Index, Neg};

use distances::Number;

mod columnar;
mod needleman_wunsch;
pub mod quality;

pub use columnar::Columnar;
pub use needleman_wunsch::{Aligner, CostMatrix};

/// The number of characters.
pub(crate) const NUM_CHARS: usize = 1 + (u8::MAX as usize);
/// The square root threshold for sub-sampling.
pub(crate) const SQRT_THRESH: usize = 1000;
/// The logarithmic threshold for sub-sampling.
pub(crate) const LOG2_THRESH: usize = 100_000;

/// A multiple sequence alignment (MSA).
pub struct Msa {
    /// The aligned sequences.
    sequences: Vec<Vec<u8>>,
    /// The gap character.
    gap: u8,
}

impl Msa {
    /// The sequences in the MSA.
    #[must_use]
    pub fn sequences(&self) -> &[Vec<u8>] {
        &self.sequences
    }

    /// The gap character.
    #[must_use]
    pub const fn gap(&self) -> u8 {
        self.gap
    }

    /// The sequences in the MSA as strings.
    #[must_use]
    pub fn strings(&self) -> Vec<String> {
        self.sequences
            .iter()
            .map(|v| String::from_utf8_lossy(v).to_string())
            .collect()
    }

    /// Create a new MSA from the columnar builder.
    #[must_use]
    pub fn from_columnar<U: Number + Neg<Output = U>>(columnar: &Columnar<U>) -> Self {
        Self {
            sequences: columnar.extract_msa(),
            gap: columnar.gap(),
        }
    }

    /// Parallel version of `from_columnar`.
    #[must_use]
    pub fn par_from_columnar<U: Number + Neg<Output = U>>(columnar: &Columnar<U>) -> Self {
        Self {
            sequences: columnar.par_extract_msa(),
            gap: columnar.gap(),
        }
    }
}

impl Index<usize> for Msa {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        &self.sequences[index]
    }
}

impl<T: AsRef<str>> From<&[T]> for Msa {
    fn from(sequences: &[T]) -> Self {
        let sequences = sequences.iter().map(|s| s.as_ref().as_bytes().to_vec()).collect();
        Self { sequences, gap: b'-' }
    }
}
