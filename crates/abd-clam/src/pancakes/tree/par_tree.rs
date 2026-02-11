//! Parallel compression and decompression of trees with items implementing the `Codec` trait.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, Tree};

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
                .filter_map(|&id| self.get_cluster(id).ok().and_then(|c| c.parent_center_index))
                .collect::<HashSet<_>>();

            // Update the recursive cost of the clusters in the current frontier.
            frontier
                .into_par_iter()
                .filter_map(|id| {
                    self.get_cluster(id)
                        .ok()
                        .and_then(|c| c.child_center_indices().map(<[_]>::to_vec))
                        .map(|child_center_indices| {
                            let centers_cost = self.par_compression_cost(id, &child_center_indices);
                            let child_costs = child_center_indices
                                .into_iter()
                                .filter_map(|cid| self.get_cluster(cid).ok().map(|c| c.annotation.1.smaller_cost()))
                                .sum::<usize>();
                            (id, centers_cost + child_costs)
                        })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|(id, cost)| {
                    if let Ok(c) = self.get_cluster_mut(id) {
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
    pub fn par_compress_all(self, min_depth: usize) -> Tree<Id, MaybeCompressed<I>, T, A, M>
    where
        T: DistanceValue,
    {
        // Annotate the clusters with their unitary and recursive compression costs and trim the tree down to the first unitary cluster along each branch.
        let mut tree = self.par_annotate_unitary_compression_costs();
        tree.par_annotate_recursive_compression_costs();
        tree.trim_to_unitary_clusters(min_depth);

        // Compress the items in the tree.
        let mut tree = tree.par_apply_to_items(&|id, item| (id, MaybeCompressed::Original(item)));
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
    fn par_annotate_unitary_compression_costs(self) -> Tree<Id, I, T, (A, CompressionCost), M>
    where
        T: Clone,
    {
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
        self.par_apply_to_items(&|id, item| {
            let item = item
                .take_original()
                .unwrap_or_else(|| unreachable!("All items should be in their original form by the time the frontier is empty"));
            (id, item)
        })
    }

    /// Parallel version of [`Self::compress_root`]
    pub fn par_compress_root(&mut self) {
        self.par_compress_subtree(0)
            .unwrap_or_else(|err| unreachable!("The center of the root cluster is never compressed. Got error: {err}"));
    }

    /// Parallel version of [`Self::decompress_root`]
    pub fn par_decompress_root(&mut self) {
        self.par_decompress_subtree(0)
            .unwrap_or_else(|err| unreachable!("The center of the root cluster is never compressed. Got error: {err}"));
    }

    /// Parallel version of [`Self::compress_subtree`]
    pub(crate) fn par_compress_subtree(&mut self, id: usize) -> Result<(), String> {
        let mut frontier = self
            .get_cluster(id)?
            .items_indices()
            .filter(|i| self.cluster_map.get(i).is_some_and(Cluster::is_leaf))
            .collect::<Vec<_>>();

        while frontier.len() > 1 {
            // The parents of the clusters in the current frontier will form the next frontier.
            let parents = frontier
                .iter()
                .filter_map(|&id| self.cluster_map.get(&id).and_then(|c| c.parent_center_index))
                .collect::<HashSet<_>>();

            let compressed_items = frontier
                .into_par_iter()
                .filter_map(|id| {
                    self.get_cluster(id)
                        .and_then(|c| {
                            let targets = c.child_center_indices().map_or_else(
                                || c.subtree_indices().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                                <[_]>::to_vec,                    // If the cluster is a parent, we only compress the child centers.
                            );
                            self.par_compressed_items(id, &targets) // Compress the items in parallel
                        })
                        .ok()
                })
                // Flatten the results and filter out the None values, which correspond to items that were already compressed.
                .flatten()
                .flatten()
                .collect::<Vec<_>>();

            // Update the compressed items in the tree.
            for (i, compressed) in compressed_items {
                self.items[i].1 = MaybeCompressed::Compressed(compressed);
            }

            // Update the frontier to the parents of the clusters in the current frontier.
            frontier = parents.into_iter().collect();
        }

        // Compress the last cluster in the frontier, which is the root of the subtree we are compressing.
        if let Some(id) = frontier.pop() {
            let compressed_items = if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
                self.par_compressed_items(id, &targets)?
            } else {
                let targets = self.get_cluster(id)?.subtree_indices().collect::<Vec<_>>();
                self.par_compressed_items(id, &targets)?
            };
            for (i, compressed) in compressed_items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Compressed(compressed);
            }
        }

        Ok(())
    }

    /// Parallel version of [`Self::decompress_subtree`]
    pub(crate) fn par_decompress_subtree(&mut self, id: usize) -> Result<(), String> {
        let mut frontier = vec![id];
        while let Some(id) = frontier.pop() {
            if let Some(child_centers) = self.par_decompress_child_centers(id)? {
                // Add the children of the cluster to the frontier because they may also be recursively compressed.
                frontier.extend(child_centers);
            } else {
                // This is a unitarily compressed cluster, so we need to decompress all the non-center items that are compressed.
                let targets = self.get_cluster(id)?.subtree_indices().collect::<Vec<_>>();
                let dec_items = self.par_decompressed_items(id, &targets)?;
                for (i, item) in dec_items.into_iter().flatten() {
                    self.items[i].1 = MaybeCompressed::Original(item);
                }
            }
        }

        Ok(())
    }

    /// Parallel version of [`Self::decompress_child_centers`]
    pub(crate) fn par_decompress_child_centers(&mut self, id: usize) -> Result<Option<Vec<usize>>, String> {
        if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
            let items = self.par_decompressed_items(id, &targets)?;
            for (i, item) in items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
            Ok(Some(targets))
        } else {
            Ok(None)
        }
    }

    /// Parallel version of [`Self::compressed_items`]
    #[expect(clippy::type_complexity)]
    pub(crate) fn par_compressed_items(&self, center: usize, targets: &[usize]) -> Result<Vec<Option<(usize, I::Compressed)>>, String> {
        let center = self.items[center]
            .1
            .original()
            .ok_or_else(|| format!("Center item at index {center} is compressed"))?;
        Ok(targets
            .par_iter()
            .map(|&i| self.items[i].1.original().map(|item| (i, center.compress(item))))
            .collect())
    }

    /// Parallel version of [`Self::decompressed_items`]
    pub(crate) fn par_decompressed_items(&self, center: usize, targets: &[usize]) -> Result<Vec<Option<(usize, I)>>, String> {
        let center = self.items[center]
            .1
            .original()
            .ok_or_else(|| format!("Center item at index {center} was compressed"))?;
        Ok(targets
            .par_iter()
            .map(|&i| self.items[i].1.compressed().map(|compressed| (i, center.decompress(compressed))))
            .collect())
    }
}
