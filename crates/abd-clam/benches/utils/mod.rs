//! Utilities for benchmarks.

mod compare_permuted;

use abd_clam::pancakes::{Decodable, Encodable};
pub use compare_permuted::compare_permuted;
use distances::number::Float;

/// A row of a tabular dataset.
#[derive(Clone)]
pub struct Row<F: Float>(Vec<F>);

impl<F: Float> Row<F> {
    /// Converts a `Row` to a vector.
    #[allow(dead_code)]
    pub fn to_vec(v: &Self) -> Vec<F> {
        v.0.clone()
    }
}

impl<F: Float> From<Vec<F>> for Row<F> {
    fn from(v: Vec<F>) -> Self {
        Self(v)
    }
}

impl<F: Float> FromIterator<F> for Row<F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<F: Float> AsRef<[F]> for Row<F> {
    fn as_ref(&self) -> &[F] {
        &self.0
    }
}

impl<F: Float> Encodable for Row<F> {
    fn as_bytes(&self) -> Box<[u8]> {
        self.0
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }

    fn encode(&self, reference: &Self) -> Box<[u8]> {
        let diffs = reference.0.iter().zip(self.0.iter()).map(|(&a, &b)| a - b).collect();
        Self::as_bytes(&diffs)
    }
}

impl<F: Float> Decodable for Row<F> {
    fn from_bytes(bytes: &[u8]) -> Self {
        bytes
            .chunks_exact(std::mem::size_of::<F>())
            .map(F::from_le_bytes)
            .collect()
    }

    fn decode(reference: &Self, bytes: &[u8]) -> Self {
        let diffs = Self::from_bytes(bytes);
        reference.0.iter().zip(diffs.0.iter()).map(|(&a, &b)| a - b).collect()
    }
}
