//! The quality measures that may be calculated on a dimension reduction.

use abd_clam::{DistanceValue, ParDataset};

use crate::{
    data::ShellData,
    metrics::{Metric, cosine, euclidean, levenshtein},
};

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
    /// Get the name of the quality measure.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Pairwise => "Pairwise Distance Distortion",
            Self::TriangleInequality => "Triangle Inequality Distortion",
            Self::Angle => "Angle Distortion",
            Self::FalseNearestNeighbors => "False Nearest Neighbors Rate",
        }
    }

    /// Measure the quality of the dimension reduction.
    pub fn measure<const DIM: usize>(
        &self,
        inp_data: &ShellData,
        metric: &Metric,
        reduced_data: &Vec<[f32; DIM]>,
        exhaustive: bool,
    ) -> f32 {
        match (inp_data, metric) {
            (ShellData::String(inp_data), Metric::Levenshtein) => {
                let data = inp_data.iter().map(|(s, _)| s).cloned().collect::<Vec<_>>();
                Self::measure_generic::<_, u32, _, _, DIM>(self, &data, &levenshtein, reduced_data, exhaustive)
            }
            (ShellData::F32(inp_data), Metric::Euclidean) => {
                Self::measure_generic::<_, f32, _, _, DIM>(self, inp_data, &euclidean, reduced_data, exhaustive)
            }
            (ShellData::F32(inp_data), Metric::Cosine) => {
                Self::measure_generic::<_, f32, _, _, DIM>(self, inp_data, &cosine, reduced_data, exhaustive)
            }
            (ShellData::F64(inp_data), Metric::Euclidean) => {
                Self::measure_generic::<_, f64, _, _, DIM>(self, inp_data, &euclidean, reduced_data, exhaustive)
            }
            (ShellData::F64(inp_data), Metric::Cosine) => {
                Self::measure_generic::<_, f64, _, _, DIM>(self, inp_data, &cosine, reduced_data, exhaustive)
            }
            _ => todo!("Implement remaining match arms"),
        }
    }

    /// Generic helper for the `measure` function.
    fn measure_generic<I, T, M, D, const DIM: usize>(
        &self,
        inp_data: &D,
        metric: &M,
        reduced_data: &Vec<[f32; DIM]>,
        exhaustive: bool,
    ) -> f32
    where
        I: Send + Sync + Clone,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        D: ParDataset<I>,
    {
        match self {
            Self::Pairwise => pairwise::measure(inp_data, metric, reduced_data, exhaustive),
            Self::TriangleInequality => triangle_inequality::measure(inp_data, metric, reduced_data, exhaustive),
            Self::Angle => angle::measure(inp_data, metric, reduced_data, exhaustive),
            Self::FalseNearestNeighbors => fnn::measure(inp_data, metric, reduced_data, exhaustive),
        }
    }
}
