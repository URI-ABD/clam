//! Save distances from polar points to all other points in each `Cluster`.

use std::path::Path;

use abd_clam::{Cluster, DistanceValue, Tree};
use ndarray::prelude::*;
use ndarray_npy::{NpzWriter, WritableElement};
use rayon::prelude::*;

use crate::trees::{ShellTree, VectorTree};

/// Save distances from polar points to all other points in each `Cluster`.
pub fn polar_distances<P: AsRef<Path>>(tree: ShellTree, out_dir: P) -> Result<(), String> {
    match tree {
        ShellTree::Lcs(tree) => save_cluster_distances(tree, out_dir),
        ShellTree::Levenshtein(tree) => save_cluster_distances(tree, out_dir),
        ShellTree::Euclidean(tree) => match tree {
            VectorTree::F32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::F64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U64(tree) => save_cluster_distances(tree, out_dir),
        },
        ShellTree::Cosine(tree) => match tree {
            VectorTree::F32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::F64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::I64(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U8(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U16(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U32(tree) => save_cluster_distances(tree, out_dir),
            VectorTree::U64(tree) => save_cluster_distances(tree, out_dir),
        },
    }
}

/// Save distances from polar points to all other points in each `Cluster`.
fn save_cluster_distances<Id, I, T, A, M, P>(tree: Tree<Id, I, T, A, M>, out_dir: P) -> Result<(), String>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + serde::Serialize + WritableElement + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    P: AsRef<Path>,
{
    let mut writer = {
        let out_path = out_dir.as_ref().join("polar_distances.npz");
        let file = std::fs::File::create(&out_path).map_err(|e| format!("Failed to create output file {out_path:?}: {e}"))?;
        NpzWriter::new_compressed(file)
    };

    let (items, cluster_map, metric) = tree.into_parts();
    let mut clusters = Vec::with_capacity(cluster_map.len());

    for mut cluster in cluster_map.values().map(Cluster::without_annotation) {
        if let Some(s) = cluster.span() {
            // We will only annotate parent clusters.

            ftlog::info!(
                "Processing cluster centered at index {} with cardinality {}",
                cluster.center_index(),
                cluster.cardinality()
            );

            let js = {
                let mut indices_range = cluster.items_indices();
                indices_range.start += 1; // Skip the center index.
                indices_range
            };

            // The left pole is the point farthest from the center.
            let center_distances = get_distances_in_range(cluster.center_index(), js.clone(), &items, &metric);
            let arg_left = {
                let (arg_max, _) = center_distances
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|| unreachable!("Cluster has no items"));
                arg_max + cluster.center_index() + 1
            };
            ftlog::info!("  Left pole index: {arg_left}");

            // The right pole is the point farthest from the left pole.
            let left_distances = get_distances_in_range(arg_left, js.clone(), &items, &metric);
            let (arg_right, polar_distance) = {
                let (arg_max, &polar_distance) = left_distances
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|| unreachable!("Cluster has no items"));
                (arg_max + cluster.center_index() + 1, polar_distance)
            };
            ftlog::info!("  Right pole index: {arg_right}");

            // Annotate the cluster with the span and polar distance.
            cluster.set_annotation((s, polar_distance));

            // Save distances from both poles to all points in the cluster.
            let left_distances = Array1::from_vec(left_distances);
            let right_distances = Array1::from_vec(get_distances_in_range(arg_right, js, &items, &metric));
            writer
                .add_array(format!("{}_l", cluster.center_index()), &left_distances)
                .map_err(|e| format!("Failed to write left distances: {e}"))?;
            writer
                .add_array(format!("{}_r", cluster.center_index()), &right_distances)
                .map_err(|e| format!("Failed to write right distances: {e}"))?;

            clusters.push(cluster);
        };
    }

    let contents = serde_json::to_string(&clusters).map_err(|e| format!("Failed to serialize clusters: {e}"))?;
    std::fs::write(out_dir.as_ref().join("clusters.json"), contents).map_err(|e| format!("Failed to write clusters file: {e}"))?;

    ftlog::info!("All clusters processed.");

    Ok(())
}

/// Computes distances from the indexed point to other points within the given range.
fn get_distances_in_range<Id, I, T, M>(i: usize, js: core::ops::Range<usize>, items: &[(Id, I)], metric: &M) -> Vec<T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    let i = &items[i].1;
    items[js].par_iter().map(|(_, j)| metric(i, j)).collect()
}
