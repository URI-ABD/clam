//! Measure the distortion of a number of triangle inequalities.

use abd_clam::{metric::ParMetric, Dataset, FlatVec};
use rand::prelude::*;

/// Measure the distortion of a number of triangle inequalities.
pub fn measure<I, M>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; 3], usize>,
    exhaustive: bool,
) -> f32
where
    I: Send + Sync,
    M: ParMetric<I, f32>,
{
    let indices = if exhaustive {
        (0..original_data.cardinality()).collect::<Vec<_>>()
    } else {
        let mut indices = (0..original_data.cardinality()).collect::<Vec<_>>();
        indices.shuffle(&mut rand::thread_rng());
        indices.truncate(1000);
        indices
    };
    measure_subsample(original_data, metric, reduced_data, &indices)
}

/// Measure the quality using a subsample of the data.
#[allow(unused_variables)]
fn measure_subsample<I, M>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; 3], usize>,
    indices: &[usize],
) -> f32
where
    I: Send + Sync,
    M: ParMetric<I, f32>,
{
    todo!()
}
