//! Measure the distortion of a number of pair-wise distances.

use abd_clam::{dataset::ParDataset, metric::ParMetric, Dataset, FlatVec};
use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

/// Measure the distortion of a number of pair-wise distances.
///
/// This is the mean relative error of the distances between pairs of points in
/// the original space and the reduced space.
pub fn measure<I, M, const DIM: usize>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; DIM], usize>,
    umap_data: &FlatVec<[f32; DIM], usize>,
    exhaustive: bool,
) -> (f32, f32)
where
    I: Send + Sync,
    M: ParMetric<I, f32>,
{
    let indices = if exhaustive {
        (0..original_data.cardinality()).collect::<Vec<_>>()
    } else {
        let mut indices = (0..original_data.cardinality()).collect::<Vec<_>>();
        indices.shuffle(&mut rand::rng());
        indices.truncate(1000);
        indices
    };
    measure_subsample(original_data, metric, reduced_data, umap_data, &indices)
}

/// Measure the quality using a subsample of the data.
fn measure_subsample<I, M, const DIM: usize>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; DIM], usize>,
    umap_data: &FlatVec<[f32; DIM], usize>,
    indices: &[usize],
) -> (f32, f32)
where
    I: Send + Sync,
    M: ParMetric<I, f32>,
{
    let original_distances = original_data.par_pairwise(indices, metric);

    let reduced_distances = reduced_data.par_pairwise(indices, &abd_clam::metric::Euclidean);
    let mbed_distortion = original_distances
        .par_iter()
        .zip(reduced_distances)
        .map(|(original, reduced)| {
            let deltas = original
                .iter()
                .zip(reduced)
                .map(|(&(_, _, o), (_, _, r))| (o, r))
                .filter(|&(o, _)| o != 0.0)
                .map(|(o, r)| r / o)
                .collect::<Vec<_>>();
            abd_clam::utils::coefficient_of_variation::<_, f32>(&deltas)
        })
        .sum::<f32>();

    let mbed_measure = mbed_distortion / indices.len().as_f32();

    let umap_distances = umap_data.par_pairwise(indices, &abd_clam::metric::Euclidean);
    let umap_distortion = original_distances
        .par_iter()
        .zip(umap_distances)
        .map(|(original, umap)| {
            let deltas = original
                .iter()
                .zip(umap)
                .map(|(&(_, _, o), (_, _, r))| (o, r))
                .filter(|&(o, _)| o != 0.0)
                .map(|(o, r)| r / o)
                .collect::<Vec<_>>();
            abd_clam::utils::coefficient_of_variation::<_, f32>(&deltas)
        })
        .sum::<f32>();
    let umap_measure = umap_distortion / indices.len().as_f32();

    (mbed_measure, umap_measure)
}
