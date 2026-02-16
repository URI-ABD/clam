//! MSA quality measure: distance distortion

use crate::{
    DistanceValue, NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

use super::{AlignedSequence, CostMatrix};

/// Distance distortion.
///
/// For a given pair of aligned sequences, the distance distortion is computed as follows:
///
/// 1. Compute the Needleman-Wunsch distance between the original (unaligned) versions of the two sequences, using the provided cost matrix.
/// 2. Compute the Hamming distance between the two aligned sequences, using the same cost matrix.
/// 3. The distance distortion for that pair of sequences is `hamming_distance / nw_distance`.
///
/// For an entire MSA, the distance distortion is the mean distance distortion across all pairs (or a representative sample of pairs) of the aligned sequences.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[must_use]
pub struct DistanceDistortion;

impl_named_algorithm_for_unit_struct!(DistanceDistortion, "distance-distortion", r"^distance-distortion$");

impl From<DistanceDistortion> for super::MeasurableAlignmentQuality {
    fn from(_: DistanceDistortion) -> Self {
        Self::DistanceDistortion
    }
}

impl From<MeasuredQuality<DistanceDistortion>> for super::AlignmentQuality {
    fn from(measured: MeasuredQuality<DistanceDistortion>) -> Self {
        Self::DistanceDistortion(measured)
    }
}

impl MeasurableQuality for DistanceDistortion {
    fn is_higher_better(&self) -> bool {
        false
    }

    fn min_possible(&self) -> f64 {
        1.0
    }

    fn max_possible(&self) -> f64 {
        f64::INFINITY
    }
}

impl<T: DistanceValue> QualityMeasurer<(&AlignedSequence, &AlignedSequence), CostMatrix<T>> for DistanceDistortion {
    fn measure_once(&self, (left, right): (&AlignedSequence, &AlignedSequence), cost_matrix: &CostMatrix<T>) -> f64 {
        let nw_distance = left.nw_distance(right, cost_matrix);
        if nw_distance == T::zero() {
            0_f64
        } else {
            let nw_distance = nw_distance
                .to_f64()
                .unwrap_or_else(|| unreachable!("DistanceValue should be convertible to f64"));
            let hamming_distance = left
                .hamming_distance(right, cost_matrix)
                .to_f64()
                .unwrap_or_else(|| unreachable!("DistanceValue should be convertible to f64"));
            hamming_distance / nw_distance
        }
    }
}
