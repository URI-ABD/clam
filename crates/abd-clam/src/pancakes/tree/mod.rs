//! Compression and decompression of trees with items implementing the `Codec` trait.

use std::collections::HashSet;

use crate::{Cluster, DistanceValue, Tree};

use super::{Codec, MaybeCompressed};

mod par_tree;

/// Stores the cost of unitary compression of a cluster and the cost of cost of compression of child centers, along with the old annotation of the cluster.
pub struct CompressionCost {
    /// The cost of compressing all non-center items in the cluster in terms of the center.
    pub unitary_cost: usize,
    /// If the cluster is a parent, the cost of recursive compression of the cluster. Otherwise, `None`.
    pub recursive_cost: Option<usize>,
}

impl CompressionCost {
    /// Creates a new `CompressionCost` with the given unitary cost and no recursive cost.
    pub const fn new(unitary_cost: usize) -> Self {
        Self {
            unitary_cost,
            recursive_cost: None,
        }
    }

    /// Returns the smaller compression cost between unitary and recursive compression. If the recursive cost is not set, returns the unitary cost.
    pub fn smaller_cost(&self) -> usize {
        self.recursive_cost.unwrap_or(self.unitary_cost).min(self.unitary_cost)
    }

    /// Returns true if recursive compression is cheaper than unitary compression, and false otherwise. If the recursive cost is not set, returns false.
    pub fn is_recursive(&self) -> bool {
        self.recursive_cost.is_some_and(|rec_cost| rec_cost < self.unitary_cost)
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, (A, CompressionCost), M>
where
    I: Codec,
{
    /// Computes the cost of recursive compression for all parent clusters in the tree and sets the recursive cost in the annotation of the clusters.
    fn annotate_recursive_compression_costs(&mut self) {
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
            for id in frontier {
                if let Some(cost) = self
                    .get_cluster(id)
                    .ok()
                    .and_then(|c| c.child_center_indices().map(<[_]>::to_vec))
                    .map(|child_center_indices| {
                        let centers_cost = self.compression_cost(id, &child_center_indices);
                        let child_costs = child_center_indices
                            .into_iter()
                            .filter_map(|cid| self.get_cluster(cid).ok().map(|c| c.annotation.1.smaller_cost()))
                            .sum::<usize>();
                        centers_cost + child_costs
                    })
                    && let Some(c) = self.get_cluster_mut(id).ok()
                {
                    c.annotation.1.recursive_cost = Some(cost);
                }
            }

            // Update the frontier to be the next frontier.
            frontier = next_frontier.into_iter().collect();
        }
    }

    /// Removes the descendants of the first unitary cluster along each branch of the tree.
    pub(crate) fn trim_to_unitary_clusters(&mut self, min_depth: usize) {
        // We start with the frontier containing the root cluster, and traverse down the tree. If we encounter a unitary cluster.
        let mut frontier = vec![0];
        let mut clusters_to_retain = HashSet::<usize>::new();

        // Expand the frontier with the children of the clusters in the frontier until we encounter unitary clusters.
        while !frontier.is_empty() {
            clusters_to_retain.extend(frontier.iter());

            frontier = frontier
                .into_iter()
                .filter_map(|id| {
                    self.get_cluster_mut(id).ok().and_then(|c| {
                        if c.depth < min_depth || c.annotation.1.is_recursive() {
                            // This cluster is either too shallow or recursively compressed, so we need to keep traversing down the tree.
                            c.child_center_indices().map(<[_]>::to_vec)
                        } else {
                            // This is a unitary compressed cluster, so we need to remove its descendants from the tree and not add them to the frontier.
                            c.children = None;
                            None
                        }
                    })
                })
                .flatten()
                .collect();
        }

        // Remove the clusters that are not in the cluster_to_retain set from the tree.
        self.cluster_map.retain(|id, _| clusters_to_retain.contains(id));
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Codec,
{
    /// Compresses the tree.
    pub fn compress_all(self, min_depth: usize) -> Tree<Id, MaybeCompressed<I>, T, A, M>
    where
        T: DistanceValue,
    {
        // Annotate the clusters with their unitary and recursive compression costs and trim the tree down to the first unitary cluster along each branch.
        let mut tree = self.annotate_unitary_compression_costs();
        tree.annotate_recursive_compression_costs();
        tree.trim_to_unitary_clusters(min_depth);

        // Compress the items in the tree.
        let mut tree = tree.apply_to_items(&|id, item| (id, MaybeCompressed::Original(item)));
        tree.compress_root();

        // Remove the cost annotations from the clusters, since they are no longer needed.
        tree.decompound_annotations().0
    }

    /// Returns the cost of compressing the indexed target items in terms of the indexed reference.
    ///
    /// # Arguments
    ///
    /// - `reference`: index of the reference item.
    /// - `targets`: indices of the target items to compress.
    fn compression_cost(&self, reference: usize, targets: &[usize]) -> usize {
        let reference = &self.items[reference].1;
        targets.iter().map(|&i| I::compressed_size(&reference.compress(&self.items[i].1))).sum()
    }

    /// Annotates all clusters in the tree with unitary compression costs.
    fn annotate_unitary_compression_costs(self) -> Tree<Id, I, T, (A, CompressionCost), M>
    where
        T: Clone,
    {
        let (items, cluster_map, metric) = self.into_parts();
        let annotator = |c: Cluster<T, A>| {
            let center = &items[c.center_index].1;
            let unitary_cost = items[c.subtree_indices()]
                .iter()
                .map(|(_, item)| I::compressed_size(&center.compress(item)))
                .sum();
            c.compound_annotation(CompressionCost::new(unitary_cost))
        };
        let cluster_map = cluster_map.into_iter().map(|(id, cluster)| (id, annotator(cluster))).collect();
        Tree::from_parts(items, cluster_map, metric)
    }
}

impl<Id, I, T, A, M> Tree<Id, MaybeCompressed<I>, T, A, M>
where
    I: Codec,
{
    /// Returns the tree with decompressed items.
    pub fn decompress_all(mut self) -> Tree<Id, I, T, A, M> {
        self.decompress_root();
        self.apply_to_items(&|id, item| {
            let item = item
                .take_original()
                .unwrap_or_else(|| unreachable!("All items should be in their original form by the time the frontier is empty"));
            (id, item)
        })
    }

    /// Compresses the tree from the root.
    pub fn compress_root(&mut self) {
        self.compress_subtree(0)
            .unwrap_or_else(|err| unreachable!("The center of the root cluster is never compressed. Got error: {err}"));
    }

    /// Decompresses the tree from the root.
    pub fn decompress_root(&mut self) {
        self.decompress_subtree(0)
            .unwrap_or_else(|err| unreachable!("The center of the root cluster is never compressed. Got error: {err}"));
    }

    /// Given the index of a cluster center, compresses the subtree of that cluster.
    ///
    /// # Arguments
    ///
    /// - `id`: index of the cluster center, which must be decompressed.
    ///
    /// # Errors
    ///
    /// - If the `id` is not the center of any cluster.
    /// - If the cluster center is compressed.
    pub(crate) fn compress_subtree(&mut self, id: usize) -> Result<(), String> {
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
                .into_iter()
                .filter_map(|id| {
                    self.get_cluster(id)
                        .and_then(|c| {
                            let targets = c.child_center_indices().map_or_else(
                                || c.subtree_indices().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                                <[_]>::to_vec,                    // If the cluster is a parent, we only compress the child centers.
                            );
                            self.compressed_items(id, &targets) // Compress the items in parallel
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
                self.compressed_items(id, &targets)?
            } else {
                let targets = self.get_cluster(id)?.subtree_indices().collect::<Vec<_>>();
                self.compressed_items(id, &targets)?
            };
            for (i, compressed) in compressed_items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Compressed(compressed);
            }
        }

        Ok(())
    }

    /// Given the index of a cluster center, decompresses the subtree of that cluster.
    ///
    /// # Arguments
    ///
    /// - `id`: index of the cluster center, which must be decompressed.
    ///
    /// # Errors
    ///
    /// - If the `id` is not the center of any cluster.
    /// - If the cluster center is compressed.
    pub(crate) fn decompress_subtree(&mut self, id: usize) -> Result<(), String> {
        let mut frontier = vec![id];
        while let Some(id) = frontier.pop() {
            let targets = if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
                // Add the children of the cluster to the frontier because they may also be recursively compressed.
                frontier.extend(&targets);
                targets
            } else {
                // This is a unitarily compressed cluster, so we need to decompress all the non-center items that are compressed.
                self.get_cluster(id)?.subtree_indices().collect()
            };
            let dec_items = self.decompressed_items(id, &targets)?;
            for (i, item) in dec_items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
        }

        Ok(())
    }

    /// Given the index of a cluster center, decompresses the child centers of that cluster.
    ///
    /// # Arguments
    ///
    /// - `id`: index of the cluster center, which must be decompressed.
    ///
    /// # Returns
    ///
    /// - If the cluster has children, returns the indices of the child centers.
    ///
    /// # Errors
    ///
    /// - If the `id` is not the center of any cluster.
    /// - If the cluster center is compressed.
    /// - If the cluster does not have children.
    pub(crate) fn decompress_child_centers(&mut self, id: usize) -> Result<Vec<usize>, String> {
        if let Some(targets) = self.get_cluster(id)?.child_center_indices().map(<[_]>::to_vec) {
            let items = self.decompressed_items(id, &targets)?;
            for (i, item) in items.into_iter().flatten() {
                self.items[i].1 = MaybeCompressed::Original(item);
            }
            Ok(targets)
        } else {
            Err(format!("Cluster with id {id} does not have child centers"))
        }
    }

    /// Returns compressed versions of the indexed items in terms of the indexed center.
    ///
    /// # Arguments
    ///
    /// - `center`: index of the center, which must currently be decompressed.
    /// - `targets`: indices of the items to compress.
    ///
    /// # Returns
    ///
    /// A vector of the same length as `targets`, where each element is `Some((index, compressed_item))` if the corresponding target was successfully
    /// compressed, and `None` if the corresponding target was already compressed.
    ///
    /// # Errors
    ///
    /// - If the indexed center is compressed.
    #[expect(clippy::type_complexity)]
    pub(crate) fn compressed_items(&self, center: usize, targets: &[usize]) -> Result<Vec<Option<(usize, I::Compressed)>>, String> {
        let center = self.items[center]
            .1
            .original()
            .ok_or_else(|| format!("Center item at index {center} is compressed"))?;
        Ok(targets
            .iter()
            .map(|&i| self.items[i].1.original().map(|item| (i, center.compress(item))))
            .collect())
    }

    /// Returns decompressed versions of the indexed items in terms of the indexed center.
    ///
    /// # Arguments
    ///
    /// - `center`: index of the center, which must currently be decompressed.
    /// - `targets`: indices of the items to decompress.
    ///
    /// # Returns
    ///
    /// A vector of the same length as `targets`, where each element is `Some((index, item))` if the corresponding target was successfully decompressed, and
    /// `None` if the corresponding target was already decompressed.
    ///
    /// # Errors
    ///
    /// - If the indexed center is compressed.
    pub(crate) fn decompressed_items(&self, center: usize, targets: &[usize]) -> Result<Vec<Option<(usize, I)>>, String> {
        let center = self.items[center]
            .1
            .original()
            .ok_or_else(|| format!("Center item at index {center} was compressed"))?;
        Ok(targets
            .iter()
            .map(|&i| self.items[i].1.compressed().map(|compressed| (i, center.decompress(compressed))))
            .collect())
    }
}
