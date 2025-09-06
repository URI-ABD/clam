//! A `Dataset` containing a multiple sequence alignment (MSA).

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use rayon::prelude::*;

use crate::{
    cakes::PermutedBall,
    musals::Columns,
    utils::{self, LOG2_THRESH, SQRT_THRESH},
    Ball, Dataset, DistanceValue, ParPartition, Partition, Permutable,
};

use super::super::{Aligner, NUM_CHARS};

/// A `Dataset` containing a multiple sequence alignment (MSA).
///
/// # Examples
///
/// ```no-run
/// // Create a cost matrix and an aligner
/// let cost_matrix = CostMatrix::extended_iupac();
/// let aligner = ???;
///
/// // Read a dataset
/// let data = ???; // FlatVec<String, String>
///
/// let metric = Levenshtein;
/// let criteria = |b: &_| true;
/// let seed = 42;
///
/// let msa = MSA::from_unaligned(&aligner, data, &metric, &criteria, Some(seed));
/// let aligned_sequences = msa.extract_sequences()?;
///
/// let out_path = ???;
/// let mut file = std::fs::File::create(out_path).unwrap();
/// for seq in aligned_sequences {
///   writeln!(file, "{seq}"));
/// }
/// // Alternatively, use the `bio` crate to write a FASTA file.
/// ```
#[derive(Clone, bitcode::Encode, bitcode::Decode, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct MSA<I: AsRef<[u8]>, T: DistanceValue, Me> {
    /// The Needleman-Wunsch aligner.
    aligner: Aligner<T>,
    /// The sequences in the MSA.
    sequences: Vec<I>,
    /// The metadata for each sequence.
    metadata: Vec<Me>,
}

impl<I: AsRef<[u8]> + FromIterator<u8>, T: DistanceValue, Me: Clone> MSA<I, T, Me> {
    /// Creates a new MSA using MUSALS from an unaligned dataset.
    ///
    /// # Arguments
    ///
    /// * `aligner` - The aligner.
    /// * `data` - The data on which to build the MSA.
    /// * `metric` - The metric used for clustering.
    /// * `criteria` - The criteria used for clustering.
    /// * `seed` - The seed to use for random number generation.
    ///
    /// # Returns
    ///
    /// The new MSA.
    ///
    /// # Errors
    ///
    /// - If any of the characters in the dataset are not `utf-8` compatible.
    pub fn from_unaligned<M: Fn(&I, &I) -> T, C: Fn(&Ball<T>) -> bool>(
        aligner: &Aligner<T>,
        mut sequences: Vec<I>,
        mut metadata: Vec<Me>,
        metric: &M,
        criteria: &C,
    ) -> Result<Self, String> {
        if sequences.len() != metadata.len() {
            return Err("The number of sequences must be equal to the number of metadata entries.".to_string());
        }

        let root = Ball::new_tree(&sequences, metric, criteria);
        let (root, permutation) = PermutedBall::from_cluster_tree(root, &mut sequences);

        metadata.permute(&permutation);
        sequences = Columns::new(b'-')
            .with_binary_tree(&root, &sequences, aligner)
            .extract_msa_rows();

        Self::from_aligned(aligner, sequences, metadata)
    }

    /// Extracts the aligned sequences from the MSA as a `Vec<String>`.
    ///
    /// # Errors
    ///
    /// - If any of the characters in the MSA are not `utf-8` compatible.
    pub fn extract_strings(&self) -> Result<Vec<String>, String> {
        self.sequences
            .iter()
            .map(|s| s.as_ref().to_vec())
            .map(String::from_utf8)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Failed to convert sequence to String: {e}"))
    }

    /// Creates the MSA object from an aligned dataset.
    ///
    /// # Arguments
    ///
    /// * `aligner` - The aligner.
    ///
    /// # Errors
    ///
    /// - If any sequence in the MSA is empty.
    /// - If the sequences in the MSA have different lengths.
    pub fn from_aligned(aligner: &Aligner<T>, sequences: Vec<I>, metadata: Vec<Me>) -> Result<Self, String> {
        if sequences.len() != metadata.len() {
            return Err("The number of sequences must be equal to the number of metadata entries.".to_string());
        }

        let (min_len, max_len) = sequences
            .iter()
            .map(|seq| seq.as_ref().len())
            .fold((usize::MAX, 0), |(min, max), len| {
                (Ord::min(min, len), Ord::max(max, len))
            });

        if min_len == 0 {
            Err("Empty sequences are not allowed in an MSA.".to_string())
        } else if min_len == max_len {
            Ok(Self {
                aligner: aligner.clone(),
                sequences,
                metadata,
            })
        } else {
            Err("Sequences in an MSA must have the same length.".to_string())
        }
    }

    /// Returns the Needleman-Wunsch aligner.
    pub const fn aligner(&self) -> &Aligner<T> {
        &self.aligner
    }

    /// Returns the sequences of the MSA.
    pub fn sequences(&self) -> &[I] {
        &self.sequences
    }

    /// Returns the metadata of the MSA.
    pub fn metadata(&self) -> &[Me] {
        &self.metadata
    }

    /// The gap character in the MSA.
    #[must_use]
    pub const fn gap(&self) -> u8 {
        self.aligner.gap()
    }

    /// Returns the width of the MSA.
    #[must_use]
    pub fn width(&self) -> usize {
        self.sequences.first().map_or(0, |s| s.as_ref().len())
    }

    /// Swaps between the row/col major order of the MSA.
    ///
    /// This will convert a row-major MSA to a col-major MSA and vice versa.
    pub fn change_major(&self) -> MSA<Vec<u8>, T, usize> {
        let rows = self.sequences().iter().map(I::as_ref).collect::<Vec<_>>();
        let cols = (0..self.width())
            .map(|i| rows.iter().map(|row| row[i]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let metadata = (0..cols.len()).collect::<Vec<_>>();
        MSA {
            aligner: self.aligner.clone(),
            sequences: cols,
            metadata,
        }
    }

    /// Computes the mean percentage of gaps in the sequences of the MSA.
    #[must_use]
    pub fn percent_gaps(&self) -> f32 {
        let num_gaps = self
            .sequences
            .iter()
            .map(AsRef::as_ref)
            .map(|s| bytecount::count(s, self.gap()))
            .collect::<Vec<_>>();
        let num_gaps: f32 = crate::utils::mean(&num_gaps);
        num_gaps / self.width() as f32
    }

    /// Scores each pair of columns in the MSA, applying a penalty for gaps and
    /// mismatches.
    ///
    /// This should only be used with col-major MSA and will give nonsensical
    /// results with row-major MSA.
    #[must_use]
    pub fn scoring_columns(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let score = self
            .sequences
            .iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, self.gap(), gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score as f32 / utils::n_pairs(self.cardinality()) as f32
    }

    /// Calculates the mean and maximum `p-distance`s of all pairwise
    /// alignments in the MSA.
    #[must_use]
    pub fn p_distance_stats(&self) -> (f32, f32) {
        let p_distances = self.p_distances();
        let n_pairs = p_distances.len();
        let (sum, max) = p_distances
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs as f32;
        (avg, max)
    }

    /// Same as `p_distance_stats`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn p_distance_stats_subsample(&self) -> (f32, f32) {
        let p_distances = self.p_distances_subsample();
        let n_pairs = p_distances.len();
        let (sum, max) = p_distances
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs as f32;
        (avg, max)
    }

    /// Calculates the `p-distance` of each pairwise alignment in the MSA.
    fn p_distances(&self) -> Vec<f32> {
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, self.gap());
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        self.apply_pairwise(&indices, scorer).collect()
    }

    /// Same as `p_distances`, but only estimates the score for a subset of the
    /// pairwise alignments.
    fn p_distances_subsample(&self) -> Vec<f32> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, self.gap());
        self.apply_pairwise(&indices, scorer).collect()
    }

    /// Scores the MSA using the distortion of the Levenshtein edit distance
    /// and the Hamming distance between each pair of sequences.
    #[must_use]
    pub fn distance_distortion(&self) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let score = self.sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, self.gap()));
        score / utils::n_pairs(self.cardinality()) as f32
    }

    /// Same as `distance_distortion`, but only estimates the score for a subset
    /// of the pairwise alignments.
    #[must_use]
    pub fn distance_distortion_subsample(&self) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH / 4, LOG2_THRESH / 4);
        let score = self.sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, self.gap()));
        score / utils::n_pairs(indices.len()) as f32
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
    pub fn scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, self.gap(), gap_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(self.cardinality()) as f32
    }

    /// Same as `scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn scoring_pairwise_subsample(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, self.gap(), gap_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(indices.len()) as f32
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
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, self.gap(), gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(self.cardinality()) as f32
    }

    /// Same as `weighted_scoring_pairwise`, but only estimates the score for a subset of
    /// the pairwise alignments.
    #[must_use]
    pub fn weighted_scoring_pairwise_subsample(
        &self,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH / 4, LOG2_THRESH / 4);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, self.gap(), gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(indices.len()) as f32
    }

    /// Applies a pairwise scorer to all pairs of sequences in the MSA.
    fn apply_pairwise<'a, F, G: DistanceValue>(&'a self, indices: &'a [usize], scorer: F) -> impl Iterator<Item = G> + 'a
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
    fn sum_of_pairs<F, G: DistanceValue>(&self, indices: &[usize], scorer: F) -> G
    where
        F: Fn(&[u8], &[u8]) -> G,
    {
        self.apply_pairwise(indices, scorer).sum()
    }
}

impl<I: AsRef<[u8]> + FromIterator<u8> + Send + Sync, T: DistanceValue + Send + Sync, Me: Clone + Send + Sync>
    MSA<I, T, Me>
{
    /// Parallel version of [`MSA::from_unaligned`](Self::from_unaligned).
    ///
    /// # Errors
    ///
    /// See [`MSA::from_unaligned`](Self::from_unaligned).
    pub fn par_from_unaligned<M: Fn(&I, &I) -> T + Send + Sync, C: (Fn(&Ball<T>) -> bool) + Send + Sync>(
        aligner: &Aligner<T>,
        mut sequences: Vec<I>,
        mut metadata: Vec<Me>,
        metric: &M,
        criteria: &C,
    ) -> Result<Self, String> {
        if sequences.len() != metadata.len() {
            return Err("The number of sequences must be equal to the number of metadata entries.".to_string());
        }

        let root = Ball::par_new_tree(&sequences, metric, criteria);
        let (root, permutation) = PermutedBall::par_from_cluster_tree(root, &mut sequences);

        metadata.permute(&permutation);
        sequences = Columns::new(b'-')
            .par_with_binary_tree(&root, &sequences, aligner)
            .par_extract_msa_rows();

        Self::from_aligned(aligner, sequences, metadata)
    }

    /// Parallel version of [`MSA::change_major`](crate::msa::dataset::msa::MSA::change_major).
    pub fn par_change_major(&self) -> MSA<Vec<u8>, T, usize> {
        let rows = self.sequences.par_iter().map(I::as_ref).collect::<Vec<_>>();
        let cols = (0..self.width())
            .into_par_iter()
            .map(|i| rows.iter().map(|row| row[i]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let metadata = (0..cols.len()).collect::<Vec<_>>();
        MSA {
            aligner: self.aligner.clone(),
            sequences: cols,
            metadata,
        }
    }

    /// Parallel version of [`MSA::scoring_columns`](crate::msa::dataset::msa::MSA::scoring_columns).
    #[must_use]
    pub fn par_scoring_columns(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let num_seqs = self.get(0).as_ref().len();
        let score = self
            .sequences
            .par_iter()
            .map(AsRef::as_ref)
            .map(|c| sc_inner(c, self.gap(), gap_penalty, mismatch_penalty))
            .sum::<usize>();
        score as f32 / utils::n_pairs(num_seqs) as f32
    }

    /// Parallel version of [`MSA::p_distance_stats`](crate::msa::dataset::msa::MSA::p_distance_stats).
    #[must_use]
    pub fn par_p_distance_stats(&self) -> (f32, f32) {
        let p_dists = self.par_p_distances();
        let n_pairs = p_dists.len();
        let (sum, max) = p_dists
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs as f32;
        (avg, max)
    }

    /// Parallel version of [`MSA::p_distance_stats_subsample`](crate::msa::dataset::msa::MSA::p_distance_stats_subsample).
    #[must_use]
    pub fn par_p_distance_stats_subsample(&self) -> (f32, f32) {
        let p_dists = self.par_p_distances_subsample();
        let n_pairs = p_dists.len();
        let (sum, max) = p_dists
            .into_iter()
            .fold((0.0, 0.0), |(sum, max), dist| (sum + dist, f32::max(max, dist)));
        let avg = sum / n_pairs as f32;
        (avg, max)
    }

    /// Parallel version of [`MSA::p_distances`](crate::msa::dataset::msa::MSA::p_distances).
    fn par_p_distances(&self) -> Vec<f32> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, self.gap());
        self.par_apply_pairwise(&indices, scorer).collect()
    }

    /// Parallel version of [`MSA::p_distances_subsample`](crate::msa::dataset::msa::MSA::p_distances_subsample).
    fn par_p_distances_subsample(&self) -> Vec<f32> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| pd_inner(s1, s2, self.gap());
        self.par_apply_pairwise(&indices, scorer).collect()
    }

    /// Parallel version of [`MSA::distance_distortion`](crate::msa::dataset::msa::MSA::distance_distortion).
    #[must_use]
    pub fn par_distance_distortion(&self) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let score = self.par_sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, self.gap()));
        score / utils::n_pairs(self.cardinality()) as f32
    }

    /// Parallel version of [`MSA::distance_distortion_subsample`](crate::msa::dataset::msa::MSA::distance_distortion_subsample).
    #[must_use]
    pub fn par_distance_distortion_subsample(&self) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH / 8, LOG2_THRESH / 8);
        let score = self.par_sum_of_pairs(&indices, |s1, s2| dd_inner(s1, s2, self.gap()));
        score / utils::n_pairs(indices.len()) as f32
    }

    /// Parallel version of [`MSA::scoring_pairwise`](crate::msa::dataset::msa::MSA::scoring_pairwise).
    #[must_use]
    pub fn par_scoring_pairwise(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, self.gap(), gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(self.cardinality()) as f32
    }

    /// Parallel version of [`MSA::scoring_pairwise_subsample`](crate::msa::dataset::msa::MSA::scoring_pairwise_subsample).
    #[must_use]
    pub fn par_scoring_pairwise_subsample(&self, gap_penalty: usize, mismatch_penalty: usize) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH, LOG2_THRESH);
        let scorer = |s1: &[u8], s2: &[u8]| sp_inner(s1, s2, self.gap(), gap_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(indices.len()) as f32
    }

    /// Parallel version of [`MSA::weighted_scoring_pairwise`](crate::msa::dataset::msa::MSA::weighted_scoring_pairwise).
    #[must_use]
    pub fn par_weighted_scoring_pairwise(
        &self,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, self.gap(), gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(self.cardinality()) as f32
    }

    /// Parallel version of [`MSA::weighted_scoring_pairwise_subsample`](crate::msa::dataset::msa::MSA::weighted_scoring_pairwise_subsample).
    #[must_use]
    pub fn par_weighted_scoring_pairwise_subsample(
        &self,
        gap_open_penalty: usize,
        gap_ext_penalty: usize,
        mismatch_penalty: usize,
    ) -> f32 {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let indices = utils::choose_samples(&indices, SQRT_THRESH, LOG2_THRESH);
        let scorer =
            |s1: &[u8], s2: &[u8]| wsp_inner(s1, s2, self.gap(), gap_open_penalty, gap_ext_penalty, mismatch_penalty);
        let score = self.par_sum_of_pairs(&indices, scorer);
        score as f32 / utils::n_pairs(indices.len()) as f32
    }

    /// Parallel version of [`MSA::apply_pairwise`](crate::msa::dataset::msa::MSA::apply_pairwise).
    fn par_apply_pairwise<'a, F, G: DistanceValue + Send + Sync>(
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
    fn par_sum_of_pairs<F, G: DistanceValue + Send + Sync>(&self, indices: &[usize], scorer: F) -> G
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
    let sz_lev = crate::utils::sz_lev_builder();
    let lev = sz_lev(&s1, &s2);

    if lev == 0 {
        1.0
    } else {
        ham as f32 / lev as f32
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
    num_mismatches as f32 / s1.len() as f32
}

impl<I: AsRef<[u8]>, T: DistanceValue, Me> AsRef<[I]> for MSA<I, T, Me> {
    fn as_ref(&self) -> &[I] {
        &self.sequences
    }
}

impl<I: AsRef<[u8]>, T: DistanceValue, Me> AsMut<[I]> for MSA<I, T, Me> {
    fn as_mut(&mut self) -> &mut [I] {
        &mut self.sequences
    }
}
