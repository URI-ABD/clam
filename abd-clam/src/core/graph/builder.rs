use crate::core::graph::EdgeSet;
use crate::{Cluster, ClusterSet, Dataset, Edge, Instance};
use distances::Number;
use std::collections::HashSet;

/// Filler function to select clusters for graph
#[allow(dead_code)]
pub fn select_clusters<U: Number>(root: &Cluster<U>, depth: usize) -> ClusterSet<U> {
    // TODO! Replace with proper cluster selection algorithm
    let mut selected_clusters = ClusterSet::new();
    for c in root.subtree() {
        if c.depth() == depth {
            selected_clusters.insert(c);
        }
    }
    selected_clusters
}

/// Detects edges between clusters based on their spatial relationships.
///
/// This function iterates through each pair of clusters in the provided `ClusterSet` and
/// checks whether there is an edge between them. An edge is formed if the distance between
/// the centers of two clusters is less than or equal to the sum of their radii.
///
/// # Arguments
///
/// * `clusters`: A reference to a `ClusterSet` containing the clusters to be analyzed.
/// * `data`: A reference to the dataset used to calculate distances between clusters.
///
/// # Returns
///
/// A `HashSet` containing the detected edges, represented by `Edge` instances.
#[allow(clippy::implicit_hasher, dead_code)]
pub fn detect_edges<'a, I: Instance, U: Number, D: Dataset<I, U>>(
    clusters: &ClusterSet<'a, U>,
    data: &D,
) -> EdgeSet<'a, U> {
    // TODO! Refactor for better performance
    // TODO! Generalize over different hashers?...
    let mut edges = HashSet::new();
    for (i, c1) in clusters.iter().enumerate() {
        for (j, c2) in clusters.iter().enumerate().skip(i + 1) {
            if i != j {
                let distance = c1.distance_to_other(data, c2);
                if distance <= c1.radius() + c2.radius() {
                    edges.insert(Edge::new(c1, c2, distance));
                }
            }
        }
    }

    edges
}
