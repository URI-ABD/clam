//! Compression and decompression of trees with items implementing the `Codec` trait.

use core::borrow::Borrow;

use std::collections::{HashMap, HashSet};

use crate::{Cluster, DistanceValue, Tree, tree::ClusterLocation};

use super::{Codec, Compressible, MaybeCompressed, PancakesTree};

mod par_tree;

/// Stores the cost of unitary compression of a cluster and the cost of cost of compression of child centers, along with the old annotation of the cluster.
#[derive(Debug, Clone)]
pub struct CompressionCost {
    /// The cost of compressing all non-center items in the cluster in terms of the center.
    unitary_cost: usize,
    /// If the cluster is a parent, the cost of recursive compression of the cluster. Otherwise, `None`.
    recursive_cost: Option<usize>,
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

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Compressible,
    T: DistanceValue,
{
    /// Annotates all clusters in the tree with unitary compression costs.
    pub(crate) fn annotate_unitary_compression_costs<C: Codec<I>>(self, codec: C) -> PancakesTree<Id, I, T, (A, CompressionCost), M, C> {
        let Self { items, metric } = self;

        let annotator = |c: &Cluster<T, A>| {
            let reference = &items[c.center_index].1;
            let unitary_cost = items[c.subtree_range()]
                .iter()
                .map(|(_, target, _)| codec.compress(reference, target))
                .map(|compressed| I::compressed_size(&compressed))
                .sum();
            CompressionCost::new(unitary_cost)
        };
        // We collect the annotations in a separate vector here because the `annotator` closure needs to borrow the items in the tree.
        let unitary_costs = items
            .iter()
            .map(|(_, _, loc)| match loc {
                ClusterLocation::Cluster(c) => Some(annotator(c)),
                ClusterLocation::CenterIndex(_) => None,
            })
            .collect::<Vec<_>>();

        // We can now take change the items in the tree to include the cluster annotations, since we are done borrowing them for the annotation step.
        let items = items
            .into_iter()
            .zip(unitary_costs)
            .map(|((id, item, loc), cost)| {
                let loc = match (loc, cost) {
                    (ClusterLocation::CenterIndex(i), _) => ClusterLocation::CenterIndex(i),
                    (ClusterLocation::Cluster(c), Some(cost)) => ClusterLocation::Cluster(c.compound_annotation(cost)),
                    _ => unreachable!("Annotation should be present for clusters"),
                };
                (id, MaybeCompressed::Original(item), loc)
            })
            .collect();

        // Construct the tree and return it.
        let tree = Tree { items, metric };
        PancakesTree { tree, codec }
    }
}

impl<Id, I, T, A, M, C> PancakesTree<Id, I, T, A, M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    /// Returns the cost of compressing the indexed target items in terms of the indexed reference.
    ///
    /// # Arguments
    ///
    /// - `reference`: index of the reference item.
    /// - `targets`: indices of the target items to compress.
    fn compression_cost(&self, reference: usize, targets: &[usize]) -> usize {
        let reference = self.items[reference]
            .1
            .original()
            .unwrap_or_else(|| unreachable!("The reference item should be decompressed when computing compression costs"));
        targets
            .iter()
            .map(|&i| match &self.tree.items[i].1 {
                MaybeCompressed::Original(target) => I::compressed_size(&self.codec.compress(reference, target)),
                MaybeCompressed::Compressed(compressed) => I::compressed_size(compressed),
            })
            .sum()
    }

    /// Returns the distance from a query item to a tree item, decompressing any required items in the process.
    pub(crate) fn distance_to_uncompressed<Query: Borrow<I>>(&mut self, query: &Query, item_index: usize) -> T
    where
        M: Fn(&I, &I) -> T,
    {
        if let Some(item) = self.items[item_index].1.original() {
            return (self.metric)(query.borrow(), item);
        }

        let path_from_root = self.path_to_item_unchecked(item_index);
        for &[l, r] in path_from_root.array_windows() {
            if let Some(r_compressed) = self.items[r].1.compressed() {
                let reference = self.items[l]
                    .1
                    .original()
                    .unwrap_or_else(|| unreachable!("All items on the path from the root to an item should be decompressed by the time we reach the item"));
                self.tree.items[r].1 = MaybeCompressed::Original(self.codec.decompress(reference, r_compressed));
            }
        }

        let item = self.items[item_index]
            .1
            .original()
            .unwrap_or_else(|| unreachable!("The target item should be decompressed by the time we reach it"));
        (self.metric)(query.borrow(), item)
    }

    /// Given the index of a cluster center, decompresses the subtree of that cluster.
    ///
    /// # Arguments
    ///
    /// - `id`: index of the cluster center, which must be decompressed.
    pub(crate) fn decompress_subtree(&mut self, id: usize) {
        let mut frontier = vec![id];
        while let Some(id) = frontier.pop()
            && let Some(cluster) = self.items[id].2.as_cluster()
        {
            let targets = cluster.child_center_indices().map_or_else(
                || cluster.subtree_range().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                |child_ids| {
                    // Add the children of the cluster to the frontier because they may also be recursively compressed.
                    frontier.extend(child_ids);
                    child_ids.to_vec()
                },
            );
            for (i, item) in self.decompressed_items(id, &targets) {
                self.tree.items[i].1 = MaybeCompressed::Original(item);
            }
        }
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
    pub(crate) fn decompress_child_centers(&mut self, id: usize) -> Option<Vec<usize>> {
        if let Some(targets) = self.get_cluster_unchecked(id).child_center_indices().map(<[_]>::to_vec) {
            for (i, item) in self.decompressed_items(id, &targets) {
                self.tree.items[i].1 = MaybeCompressed::Original(item);
            }
            Some(targets)
        } else {
            None
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
    pub(crate) fn compressed_items(&self, center: usize, targets: &[usize]) -> Vec<(usize, I::Compressed)> {
        self.items[center].1.original().map_or_else(
            // If the center is compressed, then it is impossible to have decompressed any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are decompressed and some that are compressed, so we only compress the
            // decompressed targets.
            |center| {
                targets
                    .iter()
                    .filter_map(|&i| self.tree.items[i].1.original().map(|item| (i, self.codec.compress(center, item))))
                    .collect()
            },
        )
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
    pub(crate) fn decompressed_items(&self, center: usize, targets: &[usize]) -> Vec<(usize, I)> {
        self.items[center].1.original().map_or_else(
            // If the center is compressed, we cannot decompress any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are compressed and some that are decompressed, so we only decompress the
            // compressed targets.
            |center| {
                targets
                    .iter()
                    .filter_map(|&i| self.items[i].1.compressed().map(|compressed| (i, self.codec.decompress(center, compressed))))
                    .collect()
            },
        )
    }
}

impl<Id, I, T, A, M, C> PancakesTree<Id, I, T, (A, CompressionCost), M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    /// Computes the cost of recursive compression for all parent clusters in the tree and sets the recursive cost in the annotation of the clusters.
    pub(crate) fn annotate_recursive_compression_costs(&mut self) {
        let (leaves, parents) = self.iter_clusters().partition::<Vec<_>, _>(|c| c.is_leaf());

        // The starting frontier contains all leaf clusters.
        let mut frontier = leaves.into_iter().map(|c| c.center_index).collect::<Vec<_>>();

        // The "parents-in-waiting" map stores the parent clusters whose children have not all been processed yet.
        let mut parents_in_waiting = parents
            .into_iter()
            .map(|c| (c.center_index, (c.child_center_indices().map_or(0, <[_]>::len))))
            .collect::<HashMap<_, _>>();
        let mut full_parents: HashMap<_, _>;

        // Traverse the tree from the frontier to the root, and update the recursive cost of the clusters as we go up.
        while self.root().annotation.1.recursive_cost.is_none() {
            // Update the recursive cost of the clusters in the current frontier.
            for id in frontier {
                let c = self.get_cluster_unchecked(id);
                let cost = c.child_center_indices().map_or(c.annotation.1.unitary_cost, |child_center_indices| {
                    // If the cluster is a parent, then the cost of recursive compression is the cost of compressing the child centers in terms of the center,
                    // plus the cost of compressing the children themselves.
                    let centers_cost = self.compression_cost(id, child_center_indices);
                    let child_costs = child_center_indices
                        .iter()
                        .filter_map(|&cid| self.tree.items[cid].2.as_cluster().map(|c| c.annotation.1.smaller_cost()))
                        .sum::<usize>();
                    centers_cost + child_costs
                });

                let c = self.get_cluster_unchecked_mut(id);
                c.annotation.1.recursive_cost = Some(cost);

                // Update the count of remaining children for the parent cluster.
                if let Some(pid) = c.parent_center_index
                    && let Some(count) = parents_in_waiting.get_mut(&pid)
                {
                    *count -= 1;
                }
            }

            // The next frontier contains the parents whose children have all been processed.
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|&(_, count)| count == 0);
            frontier = full_parents.into_keys().collect();
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
                    self.tree.items[id].2.as_cluster_mut().and_then(|c| {
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

        // Update the locations of the items in the retained leaves to point to the retained cluster instead of the removed clusters.
        for i in clusters_to_retain {
            if let Some(indices) = self.items[i]
                .2
                .as_cluster()
                .and_then(|c| if c.is_leaf() { Some(c.subtree_range()) } else { None })
            {
                // Only update the locations for leaf clusters.
                for j in indices {
                    self.tree.items[j].2 = ClusterLocation::CenterIndex(i);
                }
            }
        }
    }
}
