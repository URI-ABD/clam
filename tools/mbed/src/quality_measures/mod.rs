//! The quality measures that may be calculated on a dimension reduction.

use abd_clam::{metric::ParMetric, FlatVec};

mod angle;
mod pairwise;
mod triangle_inequality;

/// The quality measures that may be calculated on a dimension reduction.
#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq)]
pub enum QualityMeasures {
    /// The distortion of a number of pair-wise distances.
    #[clap(name = "pairwise")]
    Pairwise,
    /// The distortion of a number of triangle inequalities, i.e. whether the
    /// edges of triangles still have the same relative lengths.
    #[clap(name = "triangle-inequality")]
    TriangleInequality,
    /// The distortion of a number of angles between points.
    #[clap(name = "angle")]
    Angle,
}

impl QualityMeasures {
    /// Get the name of the quality measure.
    pub const fn name(&self) -> &str {
        match self {
            QualityMeasures::Pairwise => "Pairwise Distortion",
            QualityMeasures::TriangleInequality => "Triangle Inequality Distortion",
            QualityMeasures::Angle => "Angle Distortion",
        }
    }

    /// Measure the quality of the dimension reduction.
    pub fn measure<I: Send + Sync, M: ParMetric<I, f32>>(
        &self,
        original_data: &FlatVec<I, usize>,
        metric: &M,
        reduced_data: &FlatVec<[f32; 3], usize>,
        exhaustive: bool,
    ) -> f32 {
        match self {
            QualityMeasures::Pairwise => pairwise::measure(original_data, metric, reduced_data, exhaustive),
            QualityMeasures::TriangleInequality => {
                triangle_inequality::measure(original_data, metric, reduced_data, exhaustive)
            }
            QualityMeasures::Angle => angle::measure(original_data, metric, reduced_data, exhaustive),
        }
    }
}
