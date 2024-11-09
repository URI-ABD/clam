//! Quality and accuracy metrics for MSAs.

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec};

impl<U: Number, M> FlatVec<String, U, M> {
    /// Remove all gaps from all sequences in the MSA.
    #[must_use]
    pub fn remove_gaps(mut self) -> Self {
        self.instances = self
            .instances
            .into_iter()
            .map(|s| s.chars().filter(|&c| !(c == '-' || c == '.')).collect())
            .collect();
        self
    }
}

impl<U: Number, M: Send + Sync> FlatVec<String, U, M> {
    /// Parallel version of `remove_gaps`.
    #[must_use]
    pub fn par_remove_gaps(mut self) -> Self {
        self.instances = self
            .instances
            .into_par_iter()
            .map(|s| s.chars().filter(|&c| !(c == '-' || c == '.')).collect())
            .collect();
        self
    }
}

// TODO: Consider adding a new trait for MSA datasets. Then move these methods
// to that trait.

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
                    .map(|(s1, s2)| Self::_sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty))
            })
            .sum()
    }

    /// Scores a single pairwise alignment in the MSA, applying a penalty for
    /// gaps and mismatches.
    fn _sp_inner(s1: &[u8], s2: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
        s1.iter().zip(s2.iter()).fold(0, |score, (&a, &b)| {
            if a == gap_char || b == gap_char {
                score + gap_penalty
            } else if a != b {
                score + mismatch_penalty
            } else {
                score
            }
        })
    }

    /// Scores each pairwise alignment in the MSA, applying penalties for
    /// opening a gap, extending a gap, and mismatches.
    ///
    /// # Arguments
    ///
    /// * `gap_open_penalty` - The penalty for opening a gap.
    /// * `gap_ext_penalty` - The penalty for extending a gap.
    /// * `mismatch_penalty` - The penalty for a mismatch.
    ///
    /// # Returns
    ///
    /// The sum of the penalties for all pairwise alignments divided by the
    /// number of pairwise alignments.
    #[must_use]
    pub fn weighted_scoring_pairwise(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let m = self._weighted_scoring_pairwise(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `weighted_scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn weighted_scoring_pairwise_subsample(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = self.subsample_indices();
        let m = self._weighted_scoring_pairwise(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Helper function for `weighted_scoring_pairwise` and
    /// `weighted_scoring_pairwise_subsample`.
    fn _weighted_scoring_pairwise(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
        indices: &[usize],
    ) -> usize {
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
                        Self::_wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty)
                    })
            })
            .sum()
    }

    /// Scores a single pairwise alignment in the MSA, applying a penalty for
    /// opening a gap, extending a gap, and mismatches.
    fn _wsp_inner(
        s1: &[u8],
        s2: &[u8],
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> usize {
        let start = if s1[0] == gap_char || s2[0] == gap_char {
            gap_open_penalty
        } else if s1[0] != s2[0] {
            mismatch_penalty
        } else {
            0
        };

        s1.iter()
            .zip(s1.iter().skip(1))
            .zip(s2.iter().zip(s1.iter().skip(1)))
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
                    .map(|(s1, s2)| Self::_sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty))
            })
            .sum()
    }

    /// Parallel version of `weighted_scoring_pairwise`.
    #[must_use]
    pub fn par_weighted_scoring_pairwise(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let m =
            self._par_weighted_scoring_pairwise(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `weighted_scoring_pairwise_subsample`.
    #[must_use]
    pub fn par_weighted_scoring_pairwise_subsample(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = self.subsample_indices();
        let m =
            self._par_weighted_scoring_pairwise(gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty, &indices);
        m.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of `_weighted_scoring_pairwise`.
    fn _par_weighted_scoring_pairwise(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
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
                        Self::_wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty)
                    })
            })
            .sum()
    }
}
