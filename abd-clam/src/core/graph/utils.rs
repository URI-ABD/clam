use crate::{Cluster, ClusterSet, Dataset, Edge, Instance};
use distances::Number;
use std::collections::HashSet;

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
