//! The quality measures that may be calculated on a dimension reduction.

use abd_clam::{FlatVec, metric::ParMetric};

use crate::{data::ShellFlatVec, metrics::ShellMetric};

mod angle;
mod fnn;
mod pairwise;
mod triangle_inequality;

/// The quality measures that may be calculated on a dimension reduction.
#[derive(clap::ValueEnum, Debug, Clone, PartialEq, Eq)]
pub enum QualityMeasure {
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

impl QualityMeasure {
    /// Measure the quality of the dimension reduction.
    #[expect(unused_variables)]
    pub fn measure<const DIM: usize>(
        &self,
        inp_data: &ShellFlatVec,
        metric: &ShellMetric,
        reduced_data: &FlatVec<[f32; DIM], usize>,
        exhaustive: bool,
    ) -> f32 {
        todo!()
    }

    /// Generic helper for the `measure` function.
    #[expect(dead_code)]
    fn measure_generic<I: Send + Sync + Clone, M: ParMetric<I, f32>, const DIM: usize>(
        &self,
        inp_data: &FlatVec<I, usize>,
        metric: &M,
        reduced_data: &FlatVec<[f32; DIM], usize>,
        exhaustive: bool,
    ) -> f32 {
        match self {
            Self::Pairwise => pairwise::measure(inp_data, metric, reduced_data, exhaustive),
            Self::TriangleInequality => triangle_inequality::measure(inp_data, metric, reduced_data, exhaustive),
            Self::Angle => angle::measure(inp_data, metric, reduced_data, exhaustive),
            Self::FalseNearestNeighbors => fnn::measure(inp_data, metric, reduced_data, exhaustive),
        }
    }
}
