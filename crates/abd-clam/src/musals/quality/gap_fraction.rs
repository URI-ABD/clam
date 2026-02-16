//! MSA quality measure: gap fraction.

use crate::{
    NamedAlgorithm,
    utils::{MeasurableQuality, MeasuredQuality, QualityMeasurer},
};

use super::AlignedSequence;

/// Gap Fraction, which measures the fraction of positions in an aligned sequence or MSA that are gaps.
#[derive(Debug, Clone)]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct GapFraction;

impl_named_algorithm_for_unit_struct!(GapFraction, "gap-fraction", r"^gap-fraction$");

impl From<GapFraction> for super::MeasurableAlignmentQuality {
    fn from(_: GapFraction) -> Self {
        Self::GapFraction
    }
}

impl From<MeasuredQuality<GapFraction>> for super::AlignmentQuality {
    fn from(measured: MeasuredQuality<GapFraction>) -> Self {
        Self::GapFraction(measured)
    }
}

impl MeasurableQuality for GapFraction {
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

impl QualityMeasurer<&AlignedSequence, ()> for GapFraction {
    #[expect(clippy::cast_precision_loss)]
    fn measure_once(&self, seq: &AlignedSequence, (): &()) -> f64 {
        seq.gap_count() as f64 / seq.len() as f64
    }
}
