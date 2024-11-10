//! Quality and accuracy metrics for MSAs in a column-major format.

use distances::Number;

use crate::{utils, Dataset, FlatVec};

impl<T: AsRef<[u8]>, U: Number, M> FlatVec<T, U, M> {
    /// Scores each pair of columns in the MSA, applying a penalty for gaps and
    /// mismatches.
    #[must_use]
    pub fn scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let score = self.sum_of_pairs(&self.indices(), |c1, c2| {
            sc_inner(c1, c2, gap_char, gap_penalty, mismatch_penalty)
        });
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Scores each pair of columns in the MSA, applying penalties for opening a
    /// gap, extending a gap, and mismatches.
    #[must_use]
    pub fn weighted_scoring_columns(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let score = self.sum_of_pairs(&self.indices(), |c1, c2| {
            wsc_inner(c1, c2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty)
        });
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }
}

impl<T: AsRef<[u8]> + Send + Sync, U: Number, M: Send + Sync> FlatVec<T, U, M> {
    /// Parallel version of `scoring_columns`.
    #[must_use]
    pub fn par_scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let score = self.par_sum_of_pairs(&self.indices(), |c1, c2| {
            sc_inner(c1, c2, gap_char, gap_penalty, mismatch_penalty)
        });
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `weighted_scoring_columns`.
    #[must_use]
    pub fn par_weighted_scoring_columns(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let score = self.par_sum_of_pairs(&self.indices(), |c1, c2| {
            wsc_inner(c1, c2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty)
        });
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }
}

/// Scores a single pair of columns in the MSA, applying a penalty for gaps and
/// mismatches.
fn sc_inner(c1: &[u8], c2: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
    c1.iter().zip(c2.iter()).fold(0, |score, (&a, &b)| {
        if a == gap_char || b == gap_char {
            score + gap_penalty
        } else if a != b {
            score + mismatch_penalty
        } else {
            score
        }
    })
}

/// Scores a single pair of columns in the MSA, applying a penalty for opening a
/// gap, extending a gap, and mismatches.
fn wsc_inner(
    c1: &[u8],
    c2: &[u8],
    gap_char: u8,
    gap_open_penalty: usize,
    gap_ext_penalty: usize,
    mismatch_penalty: usize,
) -> usize {
    let start = if c1[0] == gap_char || c2[0] == gap_char {
        gap_open_penalty
    } else if c1[0] != c2[0] {
        mismatch_penalty
    } else {
        0
    };

    c1.iter()
        .zip(c1.iter().skip(1))
        .zip(c2.iter().zip(c2.iter().skip(1)))
        .fold(start, |score, ((&a1, &a2), (&b1, &b2))| {
            if (a2 == gap_char && a1 != gap_char) || (b2 == gap_char && b1 != gap_char) {
                score + gap_open_penalty
            } else if a2 == gap_char || b2 == gap_char {
                score + gap_ext_penalty
            } else if a2 != b2 {
                score + mismatch_penalty
            } else {
                score
            }
        })
}
