//! Measure the distortion of a number of pair-wise distances.

use abd_clam::{DistanceValue, ParDataset};
use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use crate::metrics::euclidean;

/// Measure the distortion of a number of pair-wise distances.
///
/// This is the mean relative error of the distances between pairs of points in
/// the original space and the reduced space.
pub fn measure<I, T, M, D, const DIM: usize>(
    original_data: &D,
    metric: &M,
    reduced_data: &Vec<[f32; DIM]>,
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
fn measure_subsample<I, T, M, D, const DIM: usize>(
    original_data: &D,
    metric: &M,
    reduced_data: &Vec<[f32; DIM]>,
    indices: &[usize],
) -> f32
where
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    let original_distances = original_data.par_pairwise(indices, metric);

    let reduced_distances = reduced_data.par_pairwise(indices, &euclidean::<_, _, f32>);
    let mbed_distortion = original_distances
        .par_iter()
        .zip(reduced_distances)
        .map(|(original, reduced)| {
            let deltas = original
                .iter()
                .zip(reduced)
                .map(|(&(_, _, o), (_, _, r))| (o, r))
                .filter(|&(o, _)| o != T::zero())
                .map(|(o, r)| r / o.to_f32().unwrap_or_else(|| unreachable!("Cannot convert to f32")))
                .collect::<Vec<_>>();
            abd_clam::utils::coefficient_of_variation::<_, f32>(&deltas)
        })
        .sum::<f32>();

    mbed_distortion / indices.len().as_f32()
}
