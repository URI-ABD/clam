//! Measures the FNN (False Nearest Neighbors) of a dimension reduction.

use std::collections::HashSet;

use abd_clam::{
    cakes::ParSearchAlgorithm,
    cluster::ParPartition,
    metric::{Euclidean, ParMetric},
    Ball, Dataset, FlatVec,
};
use distances::Number;
use rand::prelude::*;

/// Measure the FNN (False Nearest Neighbors) of a dimension reduction.
pub fn measure<I, M, const DIM: usize>(
    original_data: &FlatVec<I, usize>,
    metric: &M,
    reduced_data: &FlatVec<[f32; DIM], usize>,
    umap_data: &FlatVec<[f32; DIM], usize>,
    _: bool,
) -> (f32, f32)
where
    I: Send + Sync + Clone,
    M: ParMetric<I, f32>,
{
    let criteria = |_: &Ball<_>| true;
    let seed = 42;
    let depth_stride = 128;
    let original_root = Ball::par_new_tree_iterative(original_data, metric, &criteria, Some(seed), depth_stride);

    let reduced_metric = Euclidean;
    let reduced_root = Ball::par_new_tree_iterative(reduced_data, &reduced_metric, &criteria, Some(seed), depth_stride);

    let umap_root = Ball::par_new_tree_iterative(umap_data, &reduced_metric, &criteria, Some(seed), depth_stride);

    let k = 10;
    let indices = {
        let mut indices = original_data.indices().collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        indices.truncate(1000);
        indices
    };
    let original_queries = indices.iter().map(|&i| original_data[i].clone()).collect::<Vec<_>>();
    let reduced_queries = indices.iter().map(|&i| reduced_data[i]).collect::<Vec<_>>();
    let umap_queries = indices.iter().map(|&i| umap_data[i]).collect::<Vec<_>>();

    let knn_alg = abd_clam::cakes::KnnDepthFirst(k);
    let original_neighbors = knn_alg.par_batch_search(original_data, metric, &original_root, &original_queries);
    let original_neighbors = keep_indices(original_neighbors);

    let reduced_neighbors = knn_alg.par_batch_search(reduced_data, &reduced_metric, &reduced_root, &reduced_queries);
    let reduced_neighbors = keep_indices(reduced_neighbors);

    let umap_neighbors = knn_alg.par_batch_search(umap_data, &reduced_metric, &umap_root, &umap_queries);
    let umap_neighbors = keep_indices(umap_neighbors);

    let mbed_recalls = original_neighbors
        .iter()
        .zip(reduced_neighbors.iter())
        .map(|(original, reduced)| recall(original, reduced))
        .collect::<Vec<_>>();
    let mbed_recall = 1.0 - abd_clam::utils::mean::<_, f32>(&mbed_recalls);

    let umap_recalls = original_neighbors
        .iter()
        .zip(umap_neighbors.iter())
        .map(|(original, umap)| recall(original, umap))
        .collect::<Vec<_>>();
    let umap_recall = 1.0 - abd_clam::utils::mean::<_, f32>(&umap_recalls);

    (mbed_recall, umap_recall)
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
    original_neighbors.intersection(reduced_neighbors).count().as_f32() / original_neighbors.len().as_f32()
}
