//! Quality and accuracy metrics for MSAs in a row-major format.

use distances::Number;
use rayon::prelude::*;

use crate::{utils, Dataset, FlatVec, Metric};

use super::super::{LOG2_THRESH, SQRT_THRESH};

// TODO: Consider adding a new trait for MSA datasets. Then move these methods
// to that trait.

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

    /// Calculates the mean and maximum `p-distance`s of all pairwise
    /// alignments in the MSA.
    #[must_use]
    pub fn p_distance_stats(&self, gap_char: u8) -> (f32, f32) {
        let p_distances = self.p_distances(gap_char);
        let n_pairs = p_distances.len();
        let (sum, max) = p_distances
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs.as_f32();
        (avg, max)
    }

    /// Same as `p_distance_stats`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn p_distance_stats_subsample(&self, gap_char: u8) -> (f32, f32) {
        let p_distances = self.p_distances_subsample(gap_char);
        let n_pairs = p_distances.len();
        let (sum, max) = p_distances
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs.as_f32();
        (avg, max)
    }

    /// Calculates the `p-distance` of each pairwise alignment in the MSA.
    fn p_distances(&self, gap_char: u8) -> Vec<f32> {
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.apply_pairwise(&self.indices(), scorer).collect()
    }

    /// Same as `p_distances`, but only estimates the score for a subset of the
    /// pairwise alignments.
    fn p_distances_subsample(&self, gap_char: u8) -> Vec<f32> {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.apply_pairwise(&indices, scorer).collect()
    }

    /// Scores the MSA using the distortion of the Levenshtein edit distance
    /// and the Hamming distance between each pair of sequences.
    #[must_use]
    pub fn distance_distortion(&self, gap_char: u8) -> f32 {
        let score = self.sum_of_pairs(&self.indices(), |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `distance_distortion`, but only estimates the score for a subset
    /// of the pairwise alignments.
    #[must_use]
    pub fn distance_distortion_subsample(&self, gap_char: u8) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH / 4, LOG2_THRESH / 4);
        let score = self.sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
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
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&self.indices(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
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
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH / 4, LOG2_THRESH / 4);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Applies a pairwise scorer to all pairs of sequences in the MSA.
    pub(crate) fn apply_pairwise<'a, F, G: Number>(
        &'a self,
        indices: &'a [usize],
        scorer: F,
    ) -> impl Iterator<Item = G> + 'a
    where
        F: (Fn(&[u8], &[u8]) -> G) + 'a,
    {
        indices
            .iter()
            .enumerate()
            .flat_map(move |(i, &s1)| indices.iter().skip(i + 1).map(move |&s2| (s1, s2)))
            .map(|(s1, s2)| (self.get(s1).as_ref(), self.get(s2).as_ref()))
            .map(move |(s1, s2)| scorer(s1, s2))
    }

    /// Calculate the sum of the pairwise scores for a given scorer.
    pub(crate) fn sum_of_pairs<F, G: Number>(&self, indices: &[usize], scorer: F) -> G
    where
        F: Fn(&[u8], &[u8]) -> G,
    {
        self.apply_pairwise(indices, scorer).sum()
    }
}

// Parallelized implementations here
impl<T: AsRef<[u8]> + Send + Sync, U: Number, M: Send + Sync> FlatVec<T, U, M> {
    /// Calculates the average and maximum `p-distance`s of all pairwise
    /// alignments in the MSA.
    #[must_use]
    pub fn par_p_distance_stats(&self, gap_char: u8) -> (f32, f32) {
        let p_dists = self.par_p_distances(gap_char);
        let n_pairs = p_dists.len();
        let (sum, max) = p_dists
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs.as_f32();
        (avg, max)
    }

    /// Same as `par_p_distance_stats`, but only estimates the score for a
    /// subset of the pairwise alignments.
    #[must_use]
    pub fn par_p_distance_stats_subsample(&self, gap_char: u8) -> (f32, f32) {
        let p_dists = self.par_p_distances_subsample(gap_char);
        let n_pairs = p_dists.len();
        let (sum, max) = p_dists
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs.as_f32();
        (avg, max)
    }

    /// Calculates the `p-distance` of each pairwise alignment in the MSA.
    fn par_p_distances(&self, gap_char: u8) -> Vec<f32> {
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.par_apply_pairwise(&self.indices(), scorer).collect()
    }

    /// Parallel version of `p_distance_stats_subsample`.
    fn par_p_distances_subsample(&self, gap_char: u8) -> Vec<f32> {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.par_apply_pairwise(&indices, scorer).collect()
    }

    /// Parallel version of `distance_distortion`.
    #[must_use]
    pub fn par_distance_distortion(&self, gap_char: u8) -> f32 {
        let score = self.par_sum_of_pairs(&self.indices(), |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `distance_distortion_subsample`.
    #[must_use]
    pub fn par_distance_distortion_subsample(&self, gap_char: u8) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH / 8, LOG2_THRESH / 8);
        let score = self.par_sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of `scoring_pairwise`.
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&self.indices(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of `scoring_pairwise_subsample`.
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
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

    /// Parallel version of `apply_pairwise`.
    pub(crate) fn par_apply_pairwise<'a, F, G: Number>(
        &'a self,
        indices: &'a [usize],
        scorer: F,
    ) -> impl ParallelIterator<Item = G> + 'a
    where
        F: (Fn(&[u8], &[u8]) -> G) + Send + Sync + 'a,
    {
        indices
            .par_iter()
            .enumerate()
            .flat_map(move |(i, &s1)| indices.par_iter().skip(i + 1).map(move |&s2| (s1, s2)))
            .map(|(s1, s2)| (self.get(s1).as_ref(), self.get(s2).as_ref()))
            .map(move |(s1, s2)| scorer(s1, s2))
    }

    /// Calculate the sum of the pairwise scores for a given scorer.
    pub(crate) fn par_sum_of_pairs<F, G: Number>(&self, indices: &[usize], scorer: F) -> G
    where
        F: (Fn(&[u8], &[u8]) -> G) + Send + Sync,
    {
        self.par_apply_pairwise(indices, scorer).sum()
    }
}

/// Removes gap-only columns from two aligned sequences.
fn remove_gap_only_cols(s1: &[u8], s2: &[u8], gap_char: u8) -> (Vec<u8>, Vec<u8>) {
    s1.iter()
        .zip(s2.iter())
        .filter(|(&a, &b)| !(a == gap_char && b == gap_char))
        .unzip()
}

/// Scores a single pairwise alignment in the MSA, applying a penalty for
/// gaps and mismatches.
fn sp_inner(s1: &[u8], s2: &[u8], gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> usize {
    let (s1, s2) = remove_gap_only_cols(s1, s2, gap_char);
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
    let (s1, s2) = remove_gap_only_cols(s1, s2, gap_char);

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

/// Measures the distortion of the Levenshtein edit distance between the
/// unaligned sequences and the Hamming distance between the aligned sequences.
fn dd_inner(s1: &[u8], s2: &[u8], gap_char: u8) -> f32 {
    let (s1, s2) = remove_gap_only_cols(s1, s2, gap_char);
    let ham = s1.iter().zip(s2.iter()).filter(|(&a, &b)| a != b).count();

    let s1 = s1.iter().filter(|&&c| c != gap_char).copied().collect::<Vec<_>>();
    let s2 = s2.iter().filter(|&&c| c != gap_char).copied().collect::<Vec<_>>();
    let lev = stringzilla::sz::edit_distance(s1, s2);

    if lev == 0 {
        1.0
    } else {
        ham.as_f32() / lev.as_f32()
    }
}

/// Calculates the p-distance of a pair of sequences.
fn pd_inner(s1: &[u8], s2: &[u8], gap_char: u8) -> f32 {
    let (s1, s2) = remove_gap_only_cols(s1, s2, gap_char);
    let num_mismatches = s1
        .iter()
        .zip(s2.iter())
        .filter(|(&a, &b)| a != gap_char && b != gap_char && a != b)
        .count();
    num_mismatches.as_f32() / s1.len().as_f32()
}
