//! Quality and accuracy metrics for MSAs in a column-major format.

use distances::Number;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec};

impl<T: AsRef<[u8]>, U: Number, M> FlatVec<T, U, M> {
    /// Scores each pair of columns in the MSA, applying a penalty for gaps and
    /// mismatches.
    #[must_use]
    pub fn scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let score = self
            .instances
            .iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `scoring_columns`, but with a subsample of the rows.
    #[must_use]
    pub fn scoring_columns_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let num_rows = self.get(0).as_ref().len();
        let row_indices = (0..num_rows).collect::<Vec<_>>();
        let samples = utils::choose_samples(&row_indices, 1000, 100_000);
        let score = self
            .instances
            .iter()
            .map(AsRef::as_ref)
            .map(|c| samples.iter().map(|&i| c[i]).collect::<Vec<_>>())
            .map(|c| sc_inner(&c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(samples.len()).as_f32()
    }
}

impl<T: AsRef<[u8]> + Send + Sync, U: Number, M: Send + Sync> FlatVec<T, U, M> {
    /// Parallel version of `scoring_columns`.
    #[must_use]
    pub fn par_scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let num_seqs = self.get(0).as_ref().len();
        let score = self
            .instances
            .par_iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(num_seqs).as_f32()
    }

    /// Parallel version of `scoring_columns_subsample`.
    #[must_use]
    pub fn par_scoring_columns_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let num_rows = self.get(0).as_ref().len();
        let row_indices = (0..num_rows).collect::<Vec<_>>();
        let seq_ids = utils::choose_samples(&row_indices, 1000, 100_000);
        let score = self
            .instances
            .par_iter()
            .map(AsRef::as_ref)
            .map(|c| seq_ids.iter().map(|&i| c[i]).collect::<Vec<_>>())
            .map(|c| sc_inner(&c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(seq_ids.len()).as_f32()
    }
}

/// Scores a single pair of columns in the MSA, applying a penalty for gaps and
/// mismatches.
fn sc_inner(col: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
    col.iter()
        .enumerate()
        .flat_map(|(i, &a)| col.iter().skip(i + 1).map(move |&b| (a, b)))
        .fold(0, |score, (a, b)| {
            if a == gap_char || b == gap_char {
                score + gap_penalty
            } else if a != b {
                score + mismatch_penalty
            } else {
                score
            }
        })
}
