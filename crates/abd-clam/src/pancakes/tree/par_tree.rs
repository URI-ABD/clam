//! Parallel compression and decompression of trees with items implementing the `Codec` trait.

use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use crate::{Cluster, Tree};

use super::{Codec, CompressionCost, MaybeCompressed};

impl<Id, I, T, A, M> Tree<Id, I, T, (A, CompressionCost), M>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::annotate_recursive_compression_costs`]
    fn par_annotate_recursive_compression_costs(&mut self) {
        // The starting frontier contains all leaf clusters.
        let mut frontier = self
            .cluster_map
            .iter()
            .filter_map(|(&id, c)| if c.is_leaf() { Some(id) } else { None })
            .collect::<Vec<_>>();

        // Traverse the tree from the frontier to the root, and update the recursive cost of the clusters as we go up.
        while self.root().annotation.1.recursive_cost.is_none() {
            // The next frontier contains the parents of the current frontier clusters.
            let next_frontier = frontier
                .iter()
                .filter_map(|&id| self.cluster_map.get(&id).and_then(|c| c.parent_center_index))
                .collect::<HashSet<_>>();

            // Update the recursive cost of the clusters in the current frontier.
            frontier
                .into_par_iter()
                .filter_map(|id| {
                    self.cluster_map
                        .get(&id)
                        .and_then(|c| c.child_center_indices().map(<[_]>::to_vec))
                        .map(|child_center_indices| {
                            let centers_cost = self.par_compression_cost(id, &child_center_indices);
                            let child_costs = child_center_indices
                                .into_iter()
                                .filter_map(|cid| self.cluster_map.get(&cid).map(|c| c.annotation.1.smaller_cost()))
                                .sum::<usize>();
                            (id, centers_cost + child_costs)
                        })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|(id, cost)| {
                    if let Some(c) = self.cluster_map.get_mut(&id) {
                        c.annotation.1.recursive_cost = Some(cost);
                    }
                });

            // Update the frontier to be the next frontier.
            frontier = next_frontier.into_iter().collect();
        }
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::compress_all`]
    pub fn par_compress_all(self, min_depth: usize) -> Tree<Id, MaybeCompressed<I>, T, A, M> {
        // Annotate the clusters with their unitary and recursive compression costs and trim the tree down to the first unitary cluster along each branch.
        let mut tree = self.par_annotate_unitary_compression_costs();
        tree.par_annotate_recursive_compression_costs();
        tree.trim_to_unitary_clusters(min_depth);

        let (items, cluster_map, metric) = tree.into_parts();
        let items = items.into_iter().map(|(id, item)| (id, MaybeCompressed::Original(item))).collect();

        // Compress the items in the tree.
        let mut tree = Tree::from_parts(items, cluster_map, metric);
        tree.par_compress_root();

        // Remove the cost annotations from the clusters, since they are no longer needed.
        tree.decompound_annotations().0
    }

    /// Parallel version of [`Self::compression_cost`]
    fn par_compression_cost(&self, reference: usize, targets: &[usize]) -> usize {
        let reference = &self.items[reference].1;
        targets.par_iter().map(|&i| I::compressed_size(&reference.compress(&self.items[i].1))).sum()
    }

    /// Parallel version of [`Self::annotate_unitary_compression_costs`]
    fn par_annotate_unitary_compression_costs(self) -> Tree<Id, I, T, (A, CompressionCost), M> {
        let (items, cluster_map, metric) = self.into_parts();
        let annotator = |c: Cluster<T, A>| {
            let center = &items[c.center_index].1;
            let unitary_cost = items[c.subtree_indices()]
                .par_iter()
                .map(|(_, item)| I::compressed_size(&center.compress(item)))
                .sum();
            c.compound_annotation(CompressionCost::new(unitary_cost))
        };
        let cluster_map = cluster_map.into_par_iter().map(|(id, cluster)| (id, annotator(cluster))).collect();
        Tree::from_parts(items, cluster_map, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCompressed<I>, T, A, M>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::decompress_all`]
    pub fn par_decompress_all(mut self) -> Tree<Id, I, T, A, M> {
        self.par_decompress_root();
        let (items, cluster_map, metric) = self.into_parts();
        let items = items
            .into_iter()
            .map(|(id, item)| {
                let item = item
                    .take_original()
                    .unwrap_or_else(|| unreachable!("All items should be in their original form by the time the frontier is empty"));
                (id, item)
            })
            .collect();
        Tree::from_parts(items, cluster_map, metric)
    }

    /// Parallel version of [`Self::compress_root`]
    pub fn par_compress_root(&mut self) {
        let (mut frontier, parents) = self
            .cluster_map
            .par_iter()
            .map(|(&id, c)| (id, c))
            .partition::<Vec<_>, Vec<_>, _>(|(_, c)| c.is_leaf());

        let mut parents_in_waiting = parents
            .into_par_iter()
            .map(|(i, c)| (i, (c.child_center_indices().map_or(0, <[_]>::len), c)))
            .collect::<HashMap<_, _>>();

        while !frontier.is_empty() {
            for (id, c) in frontier {
                // Get the targets to compress in terms of the center.
                let targets = c.child_center_indices().map_or_else(
                    || c.subtree_indices().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                    <[_]>::to_vec,                    // If the cluster is a parent, we only compress the child centers.
                );
                // Compress the targets and overwrite the original items with the compressed ones.
                for (i, compressed) in self.par_compressed_items(id, &targets) {
                    self.items[i].1 = MaybeCompressed::Compressed(compressed);
                }
                // Update the count of remaining children for the parent cluster.
                if let Some(pid) = c.parent_center_index
                    && let Some((count, _)) = parents_in_waiting.get_mut(&pid)
                {
                    *count -= 1;
                }
            }

            // Update the frontier to the parents whose children have all been processed.
            let full_parents: HashMap<_, _>;
            (full_parents, parents_in_waiting) = parents_in_waiting.into_par_iter().partition(|&(_, (count, _))| count == 0);
            frontier = full_parents.into_iter().map(|(id, (_, cluster))| (id, cluster)).collect();
        }
    }

    /// Parallel version of [`Self::decompress_root`]
    pub fn par_decompress_root(&mut self) {
        self.par_decompress_subtree(0);
    }

    /// Parallel version of [`Self::decompress_subtree`]
    pub(crate) fn par_decompress_subtree(&mut self, id: usize) {
        let mut frontier = vec![id];
        while let Some(id) = frontier.pop()
            && let Some(cluster) = self.cluster_map.get(&id)
        {
            let targets = cluster.child_center_indices().map_or_else(
                || cluster.subtree_indices().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                |child_ids| {
                    // Add the children of the cluster to the frontier because they may also be recursively compressed.
                    frontier.extend(child_ids);
                    child_ids.to_vec()
                },
            );
            for (i, item) in self.par_decompressed_items(id, &targets) {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
        }
    }

    /// Parallel version of [`Self::decompress_child_centers`]
    pub(crate) fn par_decompress_child_centers(&mut self, id: usize) -> Result<Option<Vec<usize>>, String> {
        if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
            for (i, item) in self.par_decompressed_items(id, &targets) {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
            Ok(Some(targets))
        } else {
            Ok(None)
        }
    }

    /// Parallel version of [`Self::compressed_items`]
    pub(crate) fn par_compressed_items(&self, center: usize, targets: &[usize]) -> Vec<(usize, I::Compressed)> {
        self.items[center].1.original().map_or_else(
            // If the center is compressed, then it is impossible to have decompressed any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are decompressed and some that are compressed, so we only compress the
            // decompressed targets.
            |center| {
                targets
                    .par_iter()
                    .filter_map(|&i| self.items[i].1.original().map(|item| (i, center.compress(item))))
                    .collect()
            },
        )
    }

    /// Parallel version of [`Self::decompressed_items`]
    pub(crate) fn par_decompressed_items(&self, center: usize, targets: &[usize]) -> Vec<(usize, I)> {
        self.items[center].1.original().map_or_else(
            // If the center is compressed, we cannot decompress any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are compressed and some that are decompressed, so we only decompress the
            // compressed targets.
            |center| {
                targets
                    .par_iter()
                    .filter_map(|&i| self.items[i].1.compressed().map(|compressed| (i, center.decompress(compressed))))
                    .collect()
            },
        )
    }
}
