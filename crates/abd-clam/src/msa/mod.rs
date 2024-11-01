//! Multiple Sequence Alignment with CLAM

mod builder;
mod needleman_wunsch;
pub mod quality;

use core::ops::Index;

pub use builder::Builder;
use distances::number::Int;
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
}

impl Index<usize> for Msa {
    type Output = [u8];

    fn index(&self, index: usize) -> &Self::Output {
        &self.sequences[index]
    }
}

impl<'a, T: AsRef<[u8]>, U: Int> From<&Builder<'a, T, U>> for Msa {
    fn from(builder: &Builder<'a, T, U>) -> Self {
        let sequences = builder.extract_msa();
        Self {
            sequences,
            gap: builder.gap,
        }
    }
}
