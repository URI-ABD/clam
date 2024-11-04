//! Multiple Sequence Alignment with CLAM

mod builder;
mod needleman_wunsch;
pub mod quality;

use core::ops::{Index, Neg};

use distances::Number;

pub use builder::Builder;
pub use needleman_wunsch::{Aligner, CostMatrix};

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

    /// Create a new MSA from a builder.
    #[must_use]
    pub fn from_builder<T: AsRef<[u8]>, U: Number + Neg<Output = U>>(builder: &Builder<T, U>) -> Self {
        let sequences = builder.extract_msa();
        Self {
            sequences,
            gap: builder.aligner.gap,
        }
    }

    /// Parallel version of `from_builder`.
    #[must_use]
    pub fn par_from_builder<T: AsRef<[u8]> + Send + Sync, U: Number + Neg<Output = U>>(builder: &Builder<T, U>) -> Self {
        let sequences = builder.par_extract_msa();
        Self {
            sequences,
            gap: builder.aligner.gap,
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
