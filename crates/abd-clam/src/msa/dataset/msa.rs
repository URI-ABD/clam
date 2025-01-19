//! A `Dataset` containing a multiple sequence alignment (MSA).

use distances::Number;
use rayon::prelude::*;

use crate::{
    dataset::{AssociatesMetadata, AssociatesMetadataMut, ParDataset, Permutable},
    msa::NUM_CHARS,
    utils::{self, LOG2_THRESH, SQRT_THRESH},
    Dataset, FlatVec,
};

use super::super::Aligner;

/// A `Dataset` containing a multiple sequence alignment (MSA).
#[derive(Clone)]
#[cfg_attr(
    feature = "disk-io",
    derive(bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)
)]
pub struct MSA<I: AsRef<[u8]>, T: Number, Me> {
    /// The Needleman-Wunsch aligner.
    aligner: Aligner<T>,
    /// The data of the MSA.
    data: FlatVec<I, Me>,
    /// The name of the MSA.
    name: String,
}

impl<I: AsRef<[u8]>, T: Number, Me> Dataset<I> for MSA<I, T, Me> {
    fn name(&self) -> &str {
        &self.name
    }

    fn with_name(self, name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    fn cardinality(&self) -> usize {
        self.data.cardinality()
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.data.dimensionality_hint()
    }

    fn get(&self, index: usize) -> &I {
        self.data.get(index)
    }
}

impl<I: AsRef<[u8]> + Send + Sync, T: Number, Me: Send + Sync> ParDataset<I> for MSA<I, T, Me> {}

impl<I: AsRef<[u8]>, T: Number, Me> Permutable for MSA<I, T, Me> {
    fn permutation(&self) -> Vec<usize> {
        self.data.permutation()
    }

    fn set_permutation(&mut self, permutation: &[usize]) {
        self.data.set_permutation(permutation);
    }

    fn swap_two(&mut self, i: usize, j: usize) {
        self.data.swap_two(i, j);
    }
}

impl<I: AsRef<[u8]>, T: Number, Me> AssociatesMetadata<I, Me> for MSA<I, T, Me> {
    fn metadata(&self) -> &[Me] {
        self.data.metadata()
    }

    fn metadata_at(&self, index: usize) -> &Me {
        self.data.metadata_at(index)
    }
}

impl<I: AsRef<[u8]>, T: Number, Me, Met: Clone> AssociatesMetadataMut<I, Me, Met, MSA<I, T, Met>> for MSA<I, T, Me> {
    fn metadata_mut(&mut self) -> &mut [Me] {
        <FlatVec<I, Me> as AssociatesMetadataMut<I, Me, Met, FlatVec<I, Met>>>::metadata_mut(&mut self.data)
    }

    fn metadata_at_mut(&mut self, index: usize) -> &mut Me {
        <FlatVec<I, Me> as AssociatesMetadataMut<I, Me, Met, FlatVec<I, Met>>>::metadata_at_mut(&mut self.data, index)
    }

    fn with_metadata(self, metadata: &[Met]) -> Result<MSA<I, T, Met>, String> {
        self.data.with_metadata(metadata).map(|data| MSA {
            aligner: self.aligner,
            data,
            name: self.name,
        })
    }

    fn transform_metadata<F: Fn(&Me) -> Met>(self, f: F) -> MSA<I, T, Met> {
        MSA {
            aligner: self.aligner,
            data: self.data.transform_metadata(f),
            name: self.name,
        }
    }
}

impl<I: AsRef<[u8]>, T: Number, Me> MSA<I, T, Me> {
    /// Creates a new MSA.
    ///
    /// # Arguments
    ///
    /// * `aligner` - The Needleman-Wunsch aligner.
    /// * `data` - The data of the MSA.
    ///
    /// # Errors
    ///
    /// - If any sequence in the MSA is empty.
    /// - If the sequences in the MSA have different lengths.
    pub fn new(aligner: &Aligner<T>, data: FlatVec<I, Me>) -> Result<Self, String> {
        let (min_len, max_len) = data
            .items()
            .iter()
            .map(|item| item.as_ref().len())
            .fold((usize::MAX, 0), |(min, max), len| {
                (Ord::min(min, len), Ord::max(max, len))
            });

        if min_len == 0 {
            Err("Empty sequences are not allowed in an MSA.".to_string())
        } else if min_len == max_len {
            let name = format!("MSA({})", data.name());
            Ok(Self {
                aligner: aligner.clone(),
                data,
                name,
            })
        } else {
            Err("Sequences in an MSA must have the same length.".to_string())
        }
    }

    /// Returns the Needleman-Wunsch aligner.
    pub const fn aligner(&self) -> &Aligner<T> {
        &self.aligner
    }

    /// Returns the data of the MSA.
    pub const fn data(&self) -> &FlatVec<I, Me> {
        &self.data
    }

    /// The gap character in the MSA.
    #[must_use]
    pub const fn gap(&self) -> u8 {
        self.aligner.gap()
    }

    /// Returns the width of the MSA.
    #[must_use]
    pub fn width(&self) -> usize {
        self.dimensionality_hint().0
    }

    /// Swaps between the row/col major order of the MSA.
    ///
    /// This will convert a row-major MSA to a col-major MSA and vice versa.
    #[must_use]
    pub fn change_major(&self) -> MSA<Vec<u8>, T, usize> {
        let rows = self.data.items().iter().map(I::as_ref).collect::<Vec<_>>();
        let cols = (0..self.width())
            .map(|i| rows.iter().map(|row| row[i]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let name = format!("Columnar({})", self.name);
        let data = FlatVec::new(cols).unwrap_or_else(|e| unreachable!("Failed to create a FlatVec: {}", e));
        MSA {
            aligner: self.aligner.clone(),
            data,
            name,
        }
    }

    /// Scores each pair of columns in the MSA, applying a penalty for gaps and
    /// mismatches.
    ///
    /// This should only be used with col-major MSA and will give nonsensical
    /// results with row-major MSA.
    #[must_use]
    pub fn scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let score = self
            .data
            .items()
            .iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
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
        self.apply_pairwise(&self.indices().collect::<Vec<_>>(), scorer)
            .collect()
    }

    /// Same as `p_distances`, but only estimates the score for a subset of the
    /// pairwise alignments.
    fn p_distances_subsample(&self, gap_char: u8) -> Vec<f32> {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.apply_pairwise(&indices, scorer).collect()
    }

    /// Scores the MSA using the distortion of the Levenshtein edit distance
    /// and the Hamming distance between each pair of sequences.
    #[must_use]
    pub fn distance_distortion(&self, gap_char: u8) -> f32 {
        let score = self.sum_of_pairs(&self.indices().collect::<Vec<_>>(), |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `distance_distortion`, but only estimates the score for a subset
    /// of the pairwise alignments.
    #[must_use]
    pub fn distance_distortion_subsample(&self, gap_char: u8) -> f32 {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH / 4, LOG2_THRESH / 4);
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
        let score = self.sum_of_pairs(&self.indices().collect::<Vec<_>>(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Same as `scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH, LOG2_THRESH);
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
        let score = self.sum_of_pairs(&self.indices().collect::<Vec<_>>(), scorer);
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
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH / 4, LOG2_THRESH / 4);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Applies a pairwise scorer to all pairs of sequences in the MSA.
    fn apply_pairwise<'a, F, G: Number>(&'a self, indices: &'a [usize], scorer: F) -> impl Iterator<Item = G> + 'a
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
    fn sum_of_pairs<F, G: Number>(&self, indices: &[usize], scorer: F) -> G
    where
        F: Fn(&[u8], &[u8]) -> G,
    {
        self.apply_pairwise(indices, scorer).sum()
    }
}

impl<I: AsRef<[u8]> + Send + Sync, T: Number, Me: Send + Sync> MSA<I, T, Me> {
    /// Parallel version of [`MSA::change_major`](crate::msa::dataset::msa::MSA::change_major).
    #[must_use]
    pub fn par_change_major(&self) -> MSA<Vec<u8>, T, usize> {
        let rows = self.data.items().par_iter().map(I::as_ref).collect::<Vec<_>>();
        let cols = (0..self.width())
            .into_par_iter()
            .map(|i| rows.iter().map(|row| row[i]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let name = format!("Columnar({})", self.name);
        let data = FlatVec::new(cols).unwrap_or_else(|e| unreachable!("Failed to create a FlatVec: {}", e));
        MSA {
            aligner: self.aligner.clone(),
            data,
            name,
        }
    }

    /// Parallel version of [`MSA::scoring_columns`](crate::msa::dataset::msa::MSA::scoring_columns).
    #[must_use]
    pub fn par_scoring_columns(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let num_seqs = self.get(0).as_ref().len();
        let score = self
            .data
            .items()
            .par_iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, gap_char, gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score.as_f32() / utils::n_pairs(num_seqs).as_f32()
    }

    /// Parallel version of [`MSA::p_distance_stats`](crate::msa::dataset::msa::MSA::p_distance_stats).
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

    /// Parallel version of [`MSA::p_distance_stats_subsample`](crate::msa::dataset::msa::MSA::p_distance_stats_subsample).
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

    /// Parallel version of [`MSA::p_distances`](crate::msa::dataset::msa::MSA::p_distances).
    fn par_p_distances(&self, gap_char: u8) -> Vec<f32> {
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.par_apply_pairwise(&self.indices().collect::<Vec<_>>(), scorer)
            .collect()
    }

    /// Parallel version of [`MSA::p_distances_subsample`](crate::msa::dataset::msa::MSA::p_distances_subsample).
    fn par_p_distances_subsample(&self, gap_char: u8) -> Vec<f32> {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, gap_char);
        self.par_apply_pairwise(&indices, scorer).collect()
    }

    /// Parallel version of [`MSA::distance_distortion`](crate::msa::dataset::msa::MSA::distance_distortion).
    #[must_use]
    pub fn par_distance_distortion(&self, gap_char: u8) -> f32 {
        let score = self.par_sum_of_pairs(&self.indices().collect::<Vec<_>>(), |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of [`MSA::distance_distortion_subsample`](crate::msa::dataset::msa::MSA::distance_distortion_subsample).
    #[must_use]
    pub fn par_distance_distortion_subsample(&self, gap_char: u8) -> f32 {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH / 8, LOG2_THRESH / 8);
        let score = self.par_sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, gap_char));
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of [`MSA::scoring_pairwise`](crate::msa::dataset::msa::MSA::scoring_pairwise).
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&self.indices().collect::<Vec<_>>(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of [`MSA::scoring_pairwise_subsample`](crate::msa::dataset::msa::MSA::scoring_pairwise_subsample).
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_char: u8, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, gap_char, gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of [`MSA::weighted_scoring_pairwise`](crate::msa::dataset::msa::MSA::weighted_scoring_pairwise).
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
        let score = self.par_sum_of_pairs(&self.indices().collect::<Vec<_>>(), scorer);
        score.as_f32() / utils::n_pairs(self.cardinality()).as_f32()
    }

    /// Parallel version of [`MSA::weighted_scoring_pairwise_subsample`](crate::msa::dataset::msa::MSA::weighted_scoring_pairwise_subsample).
    #[must_use]
    pub fn par_weighted_scoring_pairwise_subsample(
        &self,
        gap_char: u8,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = utils::choose_samples(&self.indices().collect::<Vec<_>>(), SQRT_THRESH, LOG2_THRESH);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, gap_char, gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score.as_f32() / utils::n_pairs(indices.len()).as_f32()
    }

    /// Parallel version of [`MSA::apply_pairwise`](crate::msa::dataset::msa::MSA::apply_pairwise).
    fn par_apply_pairwise<'a, F, G: Number>(
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

    /// Parallel version of [`MSA::sum_of_pairs`](crate::msa::dataset::msa::MSA::sum_of_pairs).
    fn par_sum_of_pairs<F, G: Number>(&self, indices: &[usize], scorer: F) -> G
    where
        F: (Fn(&[u8], &[u8]) -> G) + Send + Sync,
    {
        self.par_apply_pairwise(indices, scorer).sum()
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
