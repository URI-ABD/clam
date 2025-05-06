//! The quality measures that may be calculated on a dimension reduction.

use abd_clam::{metric::ParMetric, FlatVec};

mod angle;
mod fnn;
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
    /// The false nearest neighbors rate.
    #[clap(name = "fnn")]
    FalseNearestNeighbors,
}

impl QualityMeasures {
    /// Get the name of the quality measure.
    pub const fn name(&self) -> &str {
        match self {
            Self::Pairwise => "Pairwise Distortion",
            Self::TriangleInequality => "Triangle Inequality Distortion",
            Self::Angle => "Angle Distortion",
            Self::FalseNearestNeighbors => "False Nearest Neighbors",
        }
    }

    /// Measure the quality of the dimension reduction.
    pub fn measure<I: Send + Sync + Clone, M: ParMetric<I, f32>, const DIM: usize>(
        &self,
        original_data: &FlatVec<I, usize>,
        metric: &M,
        reduced_data: &FlatVec<[f32; DIM], usize>,
        umap_data: &FlatVec<[f32; DIM], usize>,
        exhaustive: bool,
    ) -> (f32, f32) {
        match self {
            Self::Pairwise => pairwise::measure(original_data, metric, reduced_data, umap_data, exhaustive),
            Self::TriangleInequality => {
                triangle_inequality::measure(original_data, metric, reduced_data, umap_data, exhaustive)
            }
            Self::Angle => angle::measure(original_data, metric, reduced_data, umap_data, exhaustive),
            Self::FalseNearestNeighbors => fnn::measure(original_data, metric, reduced_data, umap_data, exhaustive),
        }
    }
}
