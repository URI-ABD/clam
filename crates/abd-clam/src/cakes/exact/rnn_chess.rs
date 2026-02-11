//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{DistanceValue, Tree};

use super::super::{ParSearch, Search};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = (tree.metric)(query, &tree.items[0].1); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while let Some((d, cluster)) = frontier.pop() {
            if self.0 + cluster.radius() < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((cluster.center_index(), d));
            }

            if d + cluster.radius() <= self.0 {
                // Fully subsumed cluster, so we can add all items in this subtree
                for (i, (_, item)) in cluster.subtree_indices().zip(&tree.items[cluster.subtree_indices()]) {
                    hits.push((i, (tree.metric)(query, item)));
                }
            } else if let Some(children) = tree.children_of(cluster) {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let overlapping_children = children.into_iter().filter_map(|child| {
                    let d_child = (tree.metric)(query, &tree.items[child.center_index()].1);
                    if d_child <= self.0 + child.radius() { Some((d_child, child)) } else { None }
                });
                frontier.extend(overlapping_children);
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                for (i, (_, item)) in cluster.subtree_indices().zip(&tree.items[cluster.subtree_indices()]) {
                    let dist = (tree.metric)(query, item);
                    if dist <= self.0 {
                        hits.push((i, dist));
                    }
                }
            }
        }

        hits
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for RnnChess<T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = (tree.metric)(query, &tree.items[0].1); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while let Some((d, cluster)) = frontier.pop() {
            if self.0 + cluster.radius() < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((cluster.center_index(), d));
            }

            if d + cluster.radius() <= self.0 {
                // Fully subsumed cluster, so we can add all items in this subtree
                for (i, (_, item)) in cluster
                    .subtree_indices()
                    .into_par_iter()
                    .zip(&tree.items[cluster.subtree_indices()])
                    .collect::<Vec<_>>()
                {
                    hits.push((i, (tree.metric)(query, item)));
                }
            } else if let Some(children) = tree.children_of(cluster) {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let overlapping_children = children
                    .into_par_iter()
                    .filter_map(|child| {
                        let d_child = (tree.metric)(query, &tree.items[child.center_index()].1);
                        if d_child <= self.0 + child.radius() { Some((d_child, child)) } else { None }
                    })
                    .collect::<Vec<_>>();
                frontier.extend(overlapping_children);
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                for (i, (_, item)) in cluster
                    .subtree_indices()
                    .into_par_iter()
                    .zip(&tree.items[cluster.subtree_indices()])
                    .collect::<Vec<_>>()
                {
                    let dist = (tree.metric)(query, item);
                    if dist <= self.0 {
                        hits.push((i, dist));
                    }
                }
            }
        }

        hits
    }
}
