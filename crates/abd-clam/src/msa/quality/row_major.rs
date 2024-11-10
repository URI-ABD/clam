//! Quality and accuracy metrics for MSAs in a row-major format.

use distances::Number;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec, Metric};

// TODO: Consider adding a new trait for MSA datasets. Then move these methods
// to that trait.

/// The square root threshold for sub-sampling.
const SQRT_THRESH: usize = 1000;
/// The logarithmic threshold for sub-sampling.
const LOG2_THRESH: usize = 100_000;

impl<T: AsRef<[u8]>, U: Number, M> FlatVec<T, U, M> {
    /// Converts the dataset into a column-major format.
    #[must_use]
    pub fn as_col_major<Tn: FromIterator<u8>>(&self, metric: Metric<Tn, U>) -> FlatVec<Tn, U, usize> {
        let rows = self.instances.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let width = rows[0].len();

        let mut instances = Vec::with_capacity(width);
        for i in 0..width {
            let col = rows.iter().map(|row| row[i]).collect();
            instances.push(col);
        }

        let dimensionality_hint = (self.cardinality(), Some(self.cardinality()));
        let name = format!("ColMajor({})", self.name);
        FlatVec {
            metric,
            instances,
            dimensionality_hint,
            permutation: (0..width).collect(),
            metadata: (0..width).collect(),
            name,
        }
    }
}

impl<T: AsRef<[u8]>, U: Number, M> FlatVec<T, U, M> {
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
        let m = self._scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &self.indices());
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
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
                    .map(|(s1, s2)| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty))
            })
            .sum()
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
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&self.indices(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
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
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Calculate the sum of the pairwise scores for a given scorer.
    pub(crate) fn sum_of_pairs<F>(&self, indices: &[usize], scorer: F) -> usize
    where
        F: Fn(&[u8], &[u8]) -> usize,
    {
        indices
            .iter()
            .map(|&i| self.get(i).as_ref())
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, self.get(j).as_ref()))
                    .map(|(s1, s2)| scorer(s1, s2))
            })
            .sum()
    }
}

// Parallelized implementations here
impl<T: AsRef<[u8]> + Send + Sync, U: Number, M: Send + Sync> FlatVec<T, U, M> {
    /// Parallel version of `scoring_pairwise`.
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let m = self._par_scoring_pairwise(gap_char, gap_penalty, mismatch_penalty, &self.indices());
        m.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `scoring_pairwise_subsample`.
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
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
                    .map(|(s1, s2)| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty))
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
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&self.indices(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
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
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Calculate the sum of the pairwise scores for a given scorer.
    pub(crate) fn par_sum_of_pairs<F>(&self, indices: &[usize], scorer: F) -> usize
    where
        F: (Fn(&[u8], &[u8]) -> usize) + Send + Sync,
    {
        indices
            .par_iter()
            .map(|&i| self.get(i).as_ref())
            .enumerate()
            .flat_map(|(i, s1)| {
                indices
                    .par_iter()
                    .skip(i + 1)
                    .map(move |&j| (s1, self.get(j).as_ref()))
                    .map(|(s1, s2)| scorer(s1, s2))
            })
            .sum()
    }
}

/// Scores a single pairwise alignment in the MSA, applying a penalty for
/// gaps and mismatches.
fn sp_inner(s1: &[u8], s2: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
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

/// Scores a single pairwise alignment in the MSA, applying a penalty for
/// opening a gap, extending a gap, and mismatches.
fn wsp_inner(
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
