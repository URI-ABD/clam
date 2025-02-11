//! Measure the quality of the dimension reduction.

use abd_clam::{metric::ParMetric, FlatVec};
use rayon::prelude::*;

use crate::quality_measures::QualityMeasures;

/// Measure the quality of the dimension reduction.
pub fn measure<I, M, const DIM: usize>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; DIM], usize>,
    quality_measures: &[QualityMeasures],
    exhaustive: bool,
) -> Vec<f32>
where
    I: Send + Sync,
    M: ParMetric<I, f32>,
{
    ftlog::info!("Measuring quality of dimension reduction...");

    quality_measures
        .par_iter()
        .map(|m| m.measure(original_data, metric, reduced_data, exhaustive))
        .collect()
}
