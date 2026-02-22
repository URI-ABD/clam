//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    Dataset, DistanceValue, Tree,
    cakes::RnnChess,
    pancakes::{Codec, MaybeCompressedItem},
};

use super::super::{CompressiveSearch, ParCompressiveSearch};

impl<Item, D, M, T, A> CompressiveSearch<Item, D, M, T, A> for RnnChess<T>
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    T: DistanceValue,
    M: Fn(&Item, &Item) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        let root_radius = tree.root().radius();
        let d_root = tree.dataset.as_slice()[0].1.distance_to_query(query, &tree.metric)?;
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root_radius {
            return Ok(Vec::new()); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.0 + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.0 {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .map(|i| tree.dataset.as_slice()[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            } else if let Ok(child_centers) = tree.decompress_child_centers(id) {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .iter()
                    .map(|&cid| {
                        tree.get_cluster(cid).map(|c| (cid, c.radius)).and_then(|(cid, radius)| {
                            tree.dataset.as_slice()[cid]
                                .1
                                .distance_to_query(query, &tree.metric)
                                .map(|d_child| (cid, d_child, radius))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.decompress_subtree(id)?;
                let distances = tree.get_cluster(id)?.subtree_indices().filter_map(|i| {
                    tree.dataset.as_slice()[i]
                        .1
                        .distance_to_query(query, &tree.metric)
                        .ok()
                        .and_then(|d| if d <= self.0 { Some((i, d)) } else { None })
                });
                hits.extend(distances);
            }
        }

        Ok(hits)
    }
}

impl<Item, D, M, T, A> ParCompressiveSearch<Item, D, M, T, A> for RnnChess<T>
where
    Item: Codec + Send + Sync,
    Item::Compressed: Send + Sync,
    D: Dataset<Item = MaybeCompressedItem<Item>> + Send + Sync,
    D::Id: Send + Sync,
    M: Fn(&Item, &Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        let root_radius = tree.root().radius();
        let d_root = tree.dataset.as_slice()[0].1.distance_to_query(query, &tree.metric)?;
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root_radius {
            return Ok(Vec::new()); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, 0, root_radius)]; // (distance to cluster center, cluster index, cluster radius)
        while let Some((d, id, radius)) = frontier.pop() {
            if self.0 + radius < d {
                continue; // No overlap
            }
            // We have some overlap, so we need to check this cluster

            if d <= self.0 {
                // Center is within query radius
                hits.push((id, d));
            }

            if d + radius <= self.0 {
                // Fully subsumed cluster, so we will decompress the subtree and add all items in this subtree to the hits
                tree.par_decompress_subtree(id)?;
                let indices = tree.get_cluster(id)?.subtree_indices();
                let distances = indices
                    .into_par_iter()
                    .map(|i| tree.dataset.as_slice()[i].1.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                    .collect::<Result<Vec<_>, _>>()?;
                hits.extend(distances);
            } else if let Some(child_centers) = tree.par_decompress_child_centers(id)? {
                // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                let distances = child_centers
                    .par_iter()
                    .map(|&cid| {
                        tree.get_cluster(cid).map(|c| (cid, c.radius)).and_then(|(cid, radius)| {
                            tree.dataset.as_slice()[cid]
                                .1
                                .distance_to_query(query, &tree.metric)
                                .map(|d_child| (cid, d_child, radius))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for &(cid, d_child, radius) in &distances {
                    if d_child <= self.0 + radius {
                        // This child cluster overlaps with the query ball, so we add it to the frontier
                        frontier.push((d_child, cid, radius));
                    }
                }
            } else {
                // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                tree.par_decompress_subtree(id)?;
                let distances = tree
                    .get_cluster(id)?
                    .subtree_indices()
                    .into_par_iter()
                    .filter_map(|i| {
                        tree.dataset.as_slice()[i]
                            .1
                            .distance_to_query(query, &tree.metric)
                            .ok()
                            .and_then(|d| if d <= self.0 { Some((i, d)) } else { None })
                    })
                    .collect::<Vec<_>>();
                hits.extend(distances);
            }
        }

        Ok(hits)
    }
}
