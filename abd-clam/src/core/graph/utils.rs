use crate::{ClusterSet, Dataset, Edge, Instance};
use distances::Number;
use std::collections::HashSet;

/// Detects edges between clusters based on their pairwise distances within the provided `clusters` set and dataset `data`.
///
/// This function computes pairwise distances between clusters and adds an edge to the result set if the distance between two clusters is
/// less than or equal to the sum of their radii. The resulting set of edges forms the connections between clusters in the graph.
///
/// # Arguments
///
/// * `clusters`: A reference to a set of clusters for which edges need to be detected.
/// * `data`: A reference to the dataset used for calculating distances between clusters.
///
/// # Returns
///
/// Returns a `HashSet` containing the detected edges between clusters based on their pairwise distances.
///
#[allow(clippy::implicit_hasher)]
pub fn detect_edges<'a, I: Instance, U: Number, D: Dataset<I, U>>(
    clusters: &ClusterSet<'a, U>,
    data: &D,
) -> HashSet<Edge<'a, U>> {
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
