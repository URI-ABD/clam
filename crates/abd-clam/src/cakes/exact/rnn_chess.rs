//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::super::{ParSearch, Search};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<T> NamedAlgorithm for RnnChess<T>
where
    T: DistanceValue,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let mut hits = Vec::new();
        let mut frontier = vec![(tree.root(), (tree.metric)(query, &tree.items[0].1))];
        let mut next_frontier = Vec::new();
        while !frontier.is_empty() {
            for (cluster, d) in frontier {
                if d <= self.0 {
                    // Center is within query radius
                    hits.push((cluster.center_index, d));
                }
                if d + cluster.radius <= self.0 {
                    // Fully subsumed cluster, so we can add all items in this subtree
                    hits.extend(cluster.subtree_indices().map(|i| (i, (tree.metric)(query, &tree.items[i].1))));
                } else if let Some(child_center_ids) = cluster.child_center_indices() {
                    // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                    next_frontier.extend(child_center_ids.iter().filter_map(|cid| {
                        tree.cluster_map.get(cid).and_then(|child| {
                            let dist = (tree.metric)(query, &tree.items[*cid].1);
                            if dist <= self.0 + child.radius { Some((child, dist)) } else { None }
                        })
                    }));
                } else {
                    // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                    hits.extend(cluster.subtree_indices().filter_map(|i| {
                        let dist = (tree.metric)(query, &tree.items[i].1);
                        if dist <= self.0 { Some((i, dist)) } else { None }
                    }));
                }
            }
            frontier = core::mem::take(&mut next_frontier);
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
        let mut hits = Vec::new();
        let mut frontier = vec![(tree.root(), (tree.metric)(query, &tree.items[0].1))];
        let mut next_frontier = Vec::new();
        while !frontier.is_empty() {
            for (cluster, d) in frontier {
                if d <= self.0 {
                    // Center is within query radius
                    hits.push((cluster.center_index, d));
                }
                if d + cluster.radius <= self.0 {
                    // Fully subsumed cluster, so we can add all items in this subtree
                    hits.par_extend(cluster.subtree_indices().into_par_iter().map(|i| (i, (tree.metric)(query, &tree.items[i].1))));
                } else if let Some(child_center_ids) = cluster.child_center_indices() {
                    // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                    next_frontier.par_extend(child_center_ids.par_iter().filter_map(|cid| {
                        tree.cluster_map.get(cid).and_then(|child| {
                            let dist = (tree.metric)(query, &tree.items[*cid].1);
                            if dist <= self.0 + child.radius { Some((child, dist)) } else { None }
                        })
                    }));
                } else {
                    // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                    hits.par_extend(cluster.subtree_indices().into_par_iter().filter_map(|i| {
                        let dist = (tree.metric)(query, &tree.items[i].1);
                        if dist <= self.0 { Some((i, dist)) } else { None }
                    }));
                }
            }
            frontier = core::mem::take(&mut next_frontier);
        }

        hits
    }
}
