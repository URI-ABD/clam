//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, cakes::RnnChess};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree};

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for RnnChess<T>
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)> {
        let root_radius = tree.root().radius();
        let d_root = tree.distance_to_uncompressed(query, 0);
        // Check to see if there is any overlap with the root
        if d_root > self.radius + root_radius {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.radius + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.radius {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.radius {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.decompress_subtree(id);
                let indices = tree.get_cluster_unchecked(id).subtree_range();
                let distances = indices
                    .map(|i| tree.items[i].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed this subtree."));
                hits.extend(distances);
            } else if let Some(child_centers) = tree.decompress_child_centers(id) {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .iter()
                    .map(|&cid| {
                        let radius = tree.get_cluster_unchecked(cid).radius;
                        tree.items[cid]
                            .1
                            .distance_to_uncompressed(query.borrow(), &tree.metric)
                            .map(|d_child| (cid, d_child, radius))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."));
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.radius + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.decompress_subtree(id);
                let distances = tree.get_cluster_unchecked(id).subtree_range().filter_map(|i| {
                    tree.items[i]
                        .1
                        .distance_to_uncompressed(query.borrow(), &tree.metric)
                        .ok()
                        .and_then(|d| if d <= self.radius { Some((i, d)) } else { None })
                });
                hits.extend(distances);
            }
        }

        hits
    }

    fn par_compressive_search<Query: Borrow<I> + Send + Sync>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        I::Compressed: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        C: Send + Sync,
    {
        let root_radius = tree.root().radius();
        let d_root = tree.distance_to_uncompressed(query, 0);
        // Check to see if there is any overlap with the root
        if d_root > self.radius + root_radius {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.radius + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.radius {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.radius {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.par_decompress_subtree(id);
                let indices = tree.get_cluster_unchecked(id).subtree_range();
                let distances = indices
                    .into_par_iter()
                    .map(|i| tree.items[i].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed this subtree."));
                hits.extend(distances);
            } else if let Some(child_centers) = tree.par_decompress_child_centers(id) {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .par_iter()
                    .map(|&cid| {
                        let radius = tree.get_cluster_unchecked(cid).radius;
                        tree.items[cid]
                            .1
                            .distance_to_uncompressed(query.borrow(), &tree.metric)
                            .map(|d_child| (cid, d_child, radius))
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."));
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.radius + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.par_decompress_subtree(id);
                let distances = tree
                    .get_cluster_unchecked(id)
                    .subtree_range()
                    .into_par_iter()
                    .filter_map(|i| {
                        tree.items[i]
                            .1
                            .distance_to_uncompressed(query.borrow(), &tree.metric)
                            .ok()
                            .and_then(|d| if d <= self.radius { Some((i, d)) } else { None })
                    })
                    .collect::<Vec<_>>();
                hits.extend(distances);
            }
        }

        hits
    }
}
