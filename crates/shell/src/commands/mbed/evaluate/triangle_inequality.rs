//! Measure the distortion of a number of triangle inequalities.

use abd_clam::{DistanceValue, ParDataset};
use rand::prelude::*;

/// Measure the distortion of a number of triangle inequalities.
pub fn measure<I, T, M, D, const DIM: usize>(
    original_data: &D,
    metric: &M,
    reduced_data: &[[f32; DIM]],
    exhaustive: bool,
) -> f32
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    let indices = if exhaustive {
        (0..original_data.cardinality()).collect::<Vec<_>>()
    } else {
        let mut indices = (0..original_data.cardinality()).collect::<Vec<_>>();
        indices.shuffle(&mut rand::rng());
        indices.truncate(1000);
        indices
    };
    measure_subsample(original_data, metric, reduced_data, &indices)
}

/// Measure the quality using a subsample of the data.
#[allow(unused_variables)]
fn measure_subsample<I, T, M, D, const DIM: usize>(
    original_data: &D,
    metric: &M,
    reduced_data: &[[f32; DIM]],
    indices: &[usize],
) -> f32
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    todo!()
}
