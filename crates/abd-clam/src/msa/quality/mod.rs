//! Quality and accuracy metrics for MSAs.

use rand::prelude::*;
use rayon::prelude::*;

use super::Msa;

impl Msa {
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
    /// The sum of the penalties for all pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize) -> usize {
        let indices = (0..self.sequences.len()).collect::<Vec<_>>();
        self._scoring_pairwise(gap_penalty, mismatch_penalty, &indices)
    }

    /// Same as `scoring_pairwise`, but only considers the sequences at the given
    /// indices.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_penalty: usize, mismatch_penalty: usize) -> usize {
        let mut indices = (0..self.sequences.len()).collect::<Vec<_>>();
        if self.sequences.len() > 10 {
            let n = crate::utils::num_samples(indices.len(), 10, 100);
            indices.shuffle(&mut rand::thread_rng());
            indices.truncate(n);
        };

        self._scoring_pairwise(gap_penalty, mismatch_penalty, &indices)
    }

    /// Helper function for `scoring_pairwise` and `scoring_pairwise_subsample`.
    fn _scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize, indices: &[usize]) -> usize {
        indices
            .iter()
            .map(|&i| &self[i])
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, &self[j]))
                    .map(|(s1, s2)| {
                        s1.iter().zip(s2.iter()).fold(0, |score, (&a, &b)| {
                            if a == self.gap || b == self.gap {
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
impl Msa {
    /// Parallelized version of `scoring_pairwise`.
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize) -> usize {
        let indices = (0..self.sequences.len()).collect::<Vec<_>>();
        self._par_scoring_pairwise(gap_penalty, mismatch_penalty, &indices)
    }

    /// Parallelized version of `scoring_pairwise_subsample`.
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_penalty: usize, mismatch_penalty: usize) -> usize {
        let mut indices = (0..self.sequences.len()).collect::<Vec<_>>();
        if self.sequences.len() > 10 {
            let n = crate::utils::num_samples(indices.len(), 10, 100);
            indices.shuffle(&mut rand::thread_rng());
            indices.truncate(n);
        };

        self._par_scoring_pairwise(gap_penalty, mismatch_penalty, &indices)
    }

    /// Parallelized version of `_scoring_pairwise`.
    fn _par_scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize, indices: &[usize]) -> usize {
        indices
            .par_iter()
            .map(|&i| &self[i])
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .par_iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, &self[j]))
                    .map(|(s1, s2)| {
                        s1.iter().zip(s2.iter()).fold(0, |score, (&a, &b)| {
                            if a == self.gap || b == self.gap {
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
