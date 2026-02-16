//! Measuring alignment quality using sum-of-pairs, column score, and other metrics.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasuredQuality, QualityMeasurer},
};

use super::{AlignedSequence, CostMatrix, MusalsTree};

mod distance_distortion;
mod fragmentation_rate;
mod gap_fraction;
mod sum_of_pairs;

pub use distance_distortion::DistanceDistortion;
pub use fragmentation_rate::FragmentationRate;
pub use gap_fraction::GapFraction;
pub use sum_of_pairs::SumOfPairs;

/// Quality metrics for sequence alignments.
#[non_exhaustive]
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[cfg_attr(feature = "shell", derive(clap::ValueEnum))]
pub enum MeasurableAlignmentQuality {
    /// Distance distortion, which measures the distortion of the metric space of the aligned sequences to that of the original sequences.
    ///
    /// This is computed as the ratio of the Hamming distance between the aligned sequences to the Needleman-Wunsch distance between the original sequences, for
    /// every pair of aligned sequences in the MSA. This relies on the intuition that a perfectly aligned pair od sequences should have the same distance in
    /// both the original and aligned metric spaces, while a poorly aligned pair of sequences will have a much larger distance in the aligned metric space than
    /// in the original metric space, due to the introduction of gaps and misalignments in the aligned sequences.
    ///
    /// Values range from 1.0 to infinity, where a score of 1.0 indicates that the alignment perfectly preserves the original distances between sequences, and
    /// higher scores indicate that the alignment introduces more distortion in the metric space of the aligned sequences as compared to the metric space of the
    /// original sequences.
    #[cfg_attr(feature = "shell", clap(name = "distance-distortion"))]
    DistanceDistortion,
    /// Fragmentation rate, which measures the ratio of the number of contiguous non-gap regions in a sequence to the total length of the sequence in the MSA.
    ///
    /// Values range from 0.0 to 1.0. A value of 0.0 indicates that there are no gaps in the sequence (with gaps still possible before or after the sequence),
    /// while a value close to 1.0 indicates that the sequence is highly fragmented with many gaps, which is generally undesirable.
    #[cfg_attr(feature = "shell", clap(name = "fragmentation-rate"))]
    FragmentationRate,
    /// Gap fraction, which measures the fraction of positions in a sequence that are gaps in the MSA.
    ///
    /// Values range from 0.0 to 1.0. A value of 0.0 means that there are no gaps in the sequence, while a value close to 1.0 indicates that the sequence is
    /// mostly gaps, which is generally undesirable.
    #[cfg_attr(feature = "shell", clap(name = "gap-fraction"))]
    GapFraction,
    /// Sum-of-pairs, which measures the fraction of mismatched aligned residue pairs in the alignment.
    ///
    /// These values range from 0.0 to 1.0, where 0.0 indicates a perfect alignment that correctly aligns all residue pairs, and 1.0 indicates an alignment that
    /// fails to align any residue pairs correctly. Lower scores indicate better alignment quality.
    #[cfg_attr(feature = "shell", clap(name = "sum-of-pairs"))]
    SumOfPairs,
}

impl MeasurableAlignmentQuality {
    /// Return a vector of all available quality metrics.
    #[must_use]
    pub fn all_variants() -> Vec<Self> {
        vec![Self::DistanceDistortion, Self::FragmentationRate, Self::GapFraction, Self::SumOfPairs]
    }
}

/// An enum to account for the types of alignment quality metrics that have been measured.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub enum AlignmentQuality {
    /// Distance distortion, which measures the distortion of the metric space of the aligned sequences to that of the original sequences.
    DistanceDistortion(MeasuredQuality<DistanceDistortion>),
    /// Fragmentation rate, which measures the ratio of the number of contiguous non-gap regions in a sequence to the total length of the sequence in the MSA.
    FragmentationRate(MeasuredQuality<FragmentationRate>),
    /// Gap fraction, which measures the fraction of positions in a sequence that are gaps in the MSA.
    GapFraction(MeasuredQuality<GapFraction>),
    /// Sum-of-pairs, which measures the fraction of correctly aligned residue pairs in the alignment.
    SumOfPairs(MeasuredQuality<SumOfPairs>),
}

impl core::fmt::Display for MeasurableAlignmentQuality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl core::fmt::Display for AlignmentQuality {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DistanceDistortion(dd) => dd.fmt(f),
            Self::FragmentationRate(fr) => fr.fmt(f),
            Self::GapFraction(gf) => gf.fmt(f),
            Self::SumOfPairs(sop) => sop.fmt(f),
        }
    }
}

impl core::str::FromStr for MeasurableAlignmentQuality {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::regex_pattern().captures(s).map_or_else(
            || Err(format!("Invalid format for MeasurableAlignmentQuality: {s}")),
            |caps| {
                let metric = caps.get(1).map(|m| m.as_str());
                match metric {
                    Some("distance-distortion") => DistanceDistortion::from_str(s).map(Self::from),
                    Some("fragmentation-rate") => FragmentationRate::from_str(s).map(Self::from),
                    Some("gap-fraction") => GapFraction::from_str(s).map(Self::from),
                    Some("sum-of-pairs") => SumOfPairs::from_str(s).map(Self::from),
                    Some(metric) => Err(format!("Unknown alignment quality metric: {metric}")),
                    None => Err(format!("Invalid format for MeasurableAlignmentQuality: {s}")),
                }
            },
        )
    }
}

impl NamedAlgorithm for MeasurableAlignmentQuality {
    fn name(&self) -> &'static str {
        match self {
            Self::FragmentationRate => FragmentationRate.name(),
            Self::DistanceDistortion => DistanceDistortion.name(),
            Self::GapFraction => GapFraction.name(),
            Self::SumOfPairs => SumOfPairs.name(),
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(r"^(distance-distortion|fragmentation-rate|gap-fraction|sum-of-pairs)$")
    }
}

impl MeasurableAlignmentQuality {
    /// Measure the specified quality metric for an aligned tree and cost matrix.
    ///
    /// # Arguments
    ///
    /// * `tree` - The aligned tree to measure the quality of.
    /// * `cost_matrix` - The cost matrix to use for distance calculations.
    /// * `sample_size` - The number of sequence pairs to sample for the measurement.
    /// * `rng` - A random number generator to use for sampling sequences.
    pub fn measure_tree<Id, T: DistanceValue, A, M, R: rand::Rng>(&self, tree: &MusalsTree<Id, T, A, M>, sample_size: usize, rng: &mut R) -> AlignmentQuality {
        let sample_indices = crate::utils::random_indices_in_range(tree.cardinality(), sample_size, rng);
        let sampled_sequences = sample_indices.iter().map(|&i| &tree.items[i].1).collect::<Vec<_>>();
        self.measure(&sampled_sequences, tree.cost_matrix())
    }

    /// Parallel version of [`Self::measure_tree`].
    pub fn par_measure_tree<Id, T: DistanceValue + Sync, A, M, R: rand::Rng>(
        &self,
        tree: &MusalsTree<Id, T, A, M>,
        sample_size: usize,
        rng: &mut R,
    ) -> AlignmentQuality {
        let sample_indices = crate::utils::random_indices_in_range(tree.cardinality(), sample_size, rng);
        let sampled_sequences = sample_indices.iter().map(|&i| &tree.items[i].1).collect::<Vec<_>>();
        self.par_measure(&sampled_sequences, tree.cost_matrix())
    }

    /// Measure the specified quality metric for the given sequences and cost matrix.
    ///
    /// It is the user's responsibility to ensure that all sequences are aligned and have the same length.
    pub fn measure<T: DistanceValue, Seq: Borrow<AlignedSequence>>(&self, sequences: &[Seq], cost_matrix: &CostMatrix<T>) -> AlignmentQuality {
        match self {
            Self::DistanceDistortion => {
                let paired_iterator = sequences
                    .iter()
                    .enumerate()
                    .flat_map(|(i, seq_i)| sequences.iter().skip(i + 1).map(move |seq_j| (seq_i.borrow(), seq_j.borrow())));
                AlignmentQuality::from(DistanceDistortion.measure_batch(paired_iterator, cost_matrix))
            }
            Self::FragmentationRate => AlignmentQuality::from(FragmentationRate.measure_batch(sequences.iter().map(Borrow::borrow), &())),
            Self::GapFraction => AlignmentQuality::from(GapFraction.measure_batch(sequences.iter().map(Borrow::borrow), &())),
            Self::SumOfPairs => {
                let columns = aligned_sequences_to_columns(sequences);
                AlignmentQuality::from(SumOfPairs.measure_batch(columns.iter().map(Vec::as_slice), cost_matrix))
            }
        }
    }

    /// Parallel version of [`Self::measure`].
    pub fn par_measure<T: DistanceValue + Sync, Seq: Borrow<AlignedSequence> + Sync>(
        &self,
        sequences: &[Seq],
        cost_matrix: &CostMatrix<T>,
    ) -> AlignmentQuality {
        match self {
            Self::DistanceDistortion => {
                let paired_iterator = sequences
                    .par_iter()
                    .enumerate()
                    .flat_map(|(i, seq_i)| sequences.par_iter().skip(i + 1).map(move |seq_j| (seq_i.borrow(), seq_j.borrow())));
                AlignmentQuality::from(DistanceDistortion.par_measure_batch(paired_iterator, cost_matrix))
            }
            Self::FragmentationRate => AlignmentQuality::from(FragmentationRate.par_measure_batch(sequences.par_iter().map(Borrow::borrow), &())),
            Self::GapFraction => AlignmentQuality::from(GapFraction.par_measure_batch(sequences.par_iter().map(Borrow::borrow), &())),
            Self::SumOfPairs => {
                let columns = aligned_sequences_to_columns(sequences);
                AlignmentQuality::from(SumOfPairs.par_measure_batch(columns.par_iter().map(Vec::as_slice), cost_matrix))
            }
        }
    }
}

/// Converts the given list of `AlignedSequence`s into a `Vec<Vec<u8>>` of columns.
///
/// It is the user's responsibility to ensure that all aligned sequences have the same length.
#[must_use]
pub fn aligned_sequences_to_columns<Seq: Borrow<AlignedSequence>>(sequences: &[Seq]) -> Vec<Vec<char>> {
    if sequences.is_empty() {
        return Vec::new();
    }

    let sequence_length = sequences[0].borrow().len();
    let mut columns = Vec::with_capacity(sequence_length);
    let mut sequence_iterators = sequences.iter().map(|seq| seq.borrow().iter()).collect::<Vec<_>>();

    for _ in 0..sequence_length {
        let column = sequence_iterators
            .iter_mut()
            .map(|it| it.next().unwrap_or_else(|| unreachable!("All aligned sequences must have the same length")))
            .collect::<Vec<_>>();
        columns.push(column);
    }

    columns
}
