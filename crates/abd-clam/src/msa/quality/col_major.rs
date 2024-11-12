//! Quality and accuracy metrics for MSAs in a column-major format.

use distances::Number;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec};

use super::super::NUM_CHARS;

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
}

/// Scores a single pair of columns in the MSA, applying a penalty for gaps and
/// mismatches.
fn sc_inner(col: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
    // Create a frequency count of the characters in the column.
    let freqs = col.iter().fold([0; NUM_CHARS], |mut freqs, &c| {
        freqs[c as usize] += 1;
        freqs
    });

    // Start scoring the column.
    let mut score = 0;

    // Calculate the number of pairs of characters of which one is a gap and
    // apply the gap penalty.
    let num_gaps = freqs[gap_char as usize];
    score += num_gaps * (col.len() - num_gaps) * gap_penalty / 2;

    // Get the frequencies of non-gap characters with non-zero frequency.
    let freqs = freqs
        .into_iter()
        .enumerate()
        .filter(|&(i, f)| (f > 0) && (i != gap_char as usize))
        .map(|(_, f)| f)
        .collect::<Vec<_>>();

    // For each combinatorial pair, add mismatch penalties.
    freqs
        .iter()
        .enumerate()
        .flat_map(|(i, &f1)| freqs.iter().skip(i + 1).map(move |&f2| (f1, f2)))
        .fold(score, |score, (f1, f2)| score + f1 * f2 * mismatch_penalty)
}
