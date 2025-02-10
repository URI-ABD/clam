//! Measure the distortion of a number of pair-wise distances.

use abd_clam::{dataset::ParDataset, metric::ParMetric, Dataset, FlatVec};
use distances::{number::Addition, Number};
use rand::prelude::*;
use rayon::prelude::*;

/// Measure the distortion of a number of pair-wise distances.
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
    let original_distances = original_data.par_pairwise(indices, metric);
    let reduced_distances = reduced_data.par_pairwise(indices, &abd_clam::metric::Euclidean);

    let distortion = original_distances
        .into_par_iter()
        .zip(reduced_distances)
        .map(|(original, reduced)| {
            let n = original.len().as_f32();
            let d = original
                .into_iter()
                .zip(reduced)
                .map(|((_, _, o), (_, _, r))| o.abs_diff(r))
                .sum::<f32>();
            d / n
        })
        .sum::<f32>();

    distortion / indices.len().as_f32()
}
