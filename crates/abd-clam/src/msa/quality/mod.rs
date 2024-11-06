//! Quality and accuracy metrics for MSAs.

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec};

impl<T: AsRef<[u8]>, U: Number, M> FlatVec<T, U, M> {
    /// Subsample some indices from for the MSA for approximating the quality
    /// metrics.
    fn subsample_indices(&self) -> Vec<usize> {
        let mut indices = (0..self.cardinality()).collect::<Vec<_>>();
        if self.cardinality() > 10_000 {
            let n = crate::utils::num_samples(indices.len(), 1000, 100_000);
            indices.shuffle(&mut rand::thread_rng());
            indices.truncate(n);
        };
        indices
    }

    /// Scores each pairwise alignment in the MSA, applying a penalty for gaps
    /// and mismatches.
    ///
    /// # Arguments
    ///
    /// * `gap_penalty` - The penalty for a gap.
    /// * `mismatch_penalty` - The penalty for a mismatch.
    ///
    /// # Returns
    ///
    /// The sum of the penalties for all pairwise alignments divided by the
    /// number of pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let m = self._scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = self.subsample_indices();
        let m = self._scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Helper function for `scoring_pairwise` and `scoring_pairwise_subsample`.
    fn _scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize, indices: &[usize]) -> usize {
        indices
            .iter()
            .map(|&i| self.get(i).as_ref())
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, self.get(j).as_ref()))
                    .map(|(s1, s2)| {
                        s1.iter().zip(s2.iter()).fold(0, |score, (&a, &b)| {
                            if a == gap_char || b == gap_char {
                                score + gap_penalty
                            } else if a != b {
                                score + mismatch_penalty
                            } else {
                                score
                            }
                        })
                    })
            })
            .sum()
    }
}

// Parallelized implementations here
impl<T: AsRef<[u8]> + Send + Sync, U: Number, M: Send + Sync> FlatVec<T, U, M> {
    /// Parallel version of `scoring_pairwise`.
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let m = self._par_scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `scoring_pairwise_subsample`.
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = self.subsample_indices();
        let m = self._par_scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of `_scoring_pairwise`.
    fn _par_scoring_pairwise(
        &self,
        gap_char: u8,
        gap_penalty: usize,
        mismatch_penalty: usize,
        indices: &[usize],
    ) -> usize {
        indices
            .par_iter()
            .map(|&i| self.get(i).as_ref())
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .par_iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, self.get(j).as_ref()))
                    .map(|(s1, s2)| {
                        s1.iter().zip(s2.iter()).fold(0, |score, (&a, &b)| {
                            if a == gap_char || b == gap_char {
                                score + gap_penalty
                            } else if a != b {
                                score + mismatch_penalty
                            } else {
                                score
                            }
                        })
                    })
            })
            .sum()
    }
}
