//! MSA quality measure: fragmentation rate.

use crate::{
    NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

use super::AlignedSequence;

/// Fragmentation Rate, measuring how "gappy" an MSA is.
///
/// The fragmentation rate of an aligned sequence is calculated as follows:
///
/// 1. Count the number of contiguous chunks of non-gap characters in the sequence. For example, the sequence `A--BC-D` has three chunks: `A`, `BC`, and `D`.
/// 2. Compute the fragmentation rate as `(number of chunks - 1) / unaligned_sequence_length`. The subtraction of 1 accounts for the fact that a sequence with
///    no gaps has one chunk, but we want to have a fragmentation rate of 0. The fragmentation rate ranges from 0 (no gaps) to 1 (every character from the
///    unaligned sequence is separated by gaps).
///
/// The fragmentation rate for an MSA is then calculated as the average fragmentation rate across all aligned sequences in the MSA. A lower fragmentation rate
/// indicates a less gappy MSA, while a higher fragmentation rate indicates a more gappy MSA. Generally, a lower fragmentation rate indicates that the sequences
/// in the MSA are more closely related to each other AND that the MSA is of higher quality.
#[derive(Debug, Clone)]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct FragmentationRate;

impl_named_algorithm_for_unit_struct!(FragmentationRate, "fragmentation-rate", r"^fragmentation-rate$");

impl From<FragmentationRate> for super::MeasurableAlignmentQuality {
    fn from(_: FragmentationRate) -> Self {
        Self::FragmentationRate
    }
}

impl From<MeasuredQuality<FragmentationRate>> for super::AlignmentQuality {
    fn from(measured: MeasuredQuality<FragmentationRate>) -> Self {
        Self::FragmentationRate(measured)
    }
}

impl MeasurableQuality for FragmentationRate {
    fn is_higher_better(&self) -> bool {
        false
    }

    fn min_possible(&self) -> f64 {
        0.0
    }

    fn max_possible(&self) -> f64 {
        1.0
    }
}

impl QualityMeasurer<&AlignedSequence, ()> for FragmentationRate {
    #[expect(clippy::cast_precision_loss)]
    fn measure_once(&self, seq: &AlignedSequence, (): &()) -> f64 {
        if seq.is_empty() {
            0.0
        } else {
            // Subtract 1 account for the fact that a sequence with no gaps has one chunk, but we want to have a chunk fraction of 0 in this case.
            let chunk_count = (seq.chunk_count() - 1) as f64;
            let seq_length = seq.len() as f64;
            chunk_count / seq_length
        }
    }
}
