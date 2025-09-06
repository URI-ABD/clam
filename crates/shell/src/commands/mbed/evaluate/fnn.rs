//! Measures the FNN (False Nearest Neighbors) of a dimension reduction.

use std::collections::HashSet;

use abd_clam::{Ball, DistanceValue, ParDataset, ParPartition, cakes::ParSearchAlgorithm};
use rand::prelude::*;

use crate::metrics::euclidean;

/// Measure the FNN (False Nearest Neighbors) of a dimension reduction.
pub fn measure<I, T, M, D, const DIM: usize>(
    original_data: &D,
    metric: &M,
    reduced_data: &Vec<[f32; DIM]>,
    _: bool,
) -> f32
where
    I: Send + Sync + Clone,
    T: DistanceValue + Send + Sync,
    M: (Fn(&I, &I) -> T) + Send + Sync,
    D: ParDataset<I>,
{
    let seed = 42;
    let depth_stride = 128;

    let criteria = |_: &Ball<_>| true;
    let original_root = Ball::par_new_tree_iterative(original_data, metric, &criteria, depth_stride);

    let reduced_metric = euclidean::<_, _, f32>;
    let criteria = |_: &Ball<_>| true;
    let reduced_root = Ball::par_new_tree_iterative(reduced_data, &reduced_metric, &criteria, depth_stride);

    let k = 10;
    let indices = {
        let mut indices = (0..original_data.cardinality()).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        indices.truncate(1000);
        indices
    };
    let original_queries = indices
        .iter()
        .map(|&i| original_data.get(i).clone())
        .collect::<Vec<_>>();
    let reduced_queries = indices.iter().map(|&i| reduced_data[i]).collect::<Vec<_>>();

    let knn_alg = abd_clam::cakes::KnnDepthFirst(k);
    let original_neighbors = knn_alg.par_batch_search(original_data, metric, &original_root, &original_queries);
    let original_neighbors = keep_indices(original_neighbors);

    let reduced_neighbors = knn_alg.par_batch_search(reduced_data, &reduced_metric, &reduced_root, &reduced_queries);
    let reduced_neighbors = keep_indices(reduced_neighbors);

    let mbed_recalls = original_neighbors
        .iter()
        .zip(reduced_neighbors.iter())
        .map(|(original, reduced)| recall(original, reduced))
        .collect::<Vec<_>>();

    1.0 - abd_clam::utils::mean::<_, f32>(&mbed_recalls)
}

/// Keep only the indices of the neighbors and convert the inner Vector to a
/// `HashSet`.
fn keep_indices<F>(values: Vec<Vec<(usize, F)>>) -> Vec<HashSet<usize>> {
    values
        .into_iter()
        .map(|row| row.into_iter().map(|(i, _)| i).collect::<HashSet<_>>())
        .collect::<Vec<_>>()
}

/// Calculate the recall of the lists of neighbors.
fn recall(original_neighbors: &HashSet<usize>, reduced_neighbors: &HashSet<usize>) -> f32 {
    original_neighbors.intersection(reduced_neighbors).count() as f32 / original_neighbors.len() as f32
}
