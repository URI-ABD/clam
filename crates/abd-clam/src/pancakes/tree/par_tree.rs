//! Parallel compression and decompression of trees with items implementing the `Codec` trait.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{Cluster, Tree, tree::ClusterLocation};

use super::{Codec, Compressible, CompressionCost, MaybeCompressed, PancakesTree};

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::annotate_unitary_compression_costs`]
    pub(crate) fn par_annotate_unitary_compression_costs<C: Codec<I> + Send + Sync>(self, codec: C) -> PancakesTree<Id, I, T, (A, CompressionCost), M, C> {
        let Self { items, metric } = self;

        let annotator = |c: &Cluster<T, A>| {
            let reference = &items[c.center_index].1;
            let unitary_cost = items[c.subtree_range()]
                .par_iter()
                .map(|(_, target, _)| codec.compress(reference, target))
                .map(|compressed| I::compressed_size(&compressed))
                .sum();
            CompressionCost::new(unitary_cost)
        };
        // We collect the annotations in a separate vector here because the `annotator` closure needs to borrow the items in the tree.
        let unitary_costs = items
            .par_iter()
            .map(|(_, _, loc)| match loc {
                ClusterLocation::Cluster(c) => Some(annotator(c)),
                ClusterLocation::CenterIndex(_) => None,
            })
            .collect::<Vec<_>>();

        // We can now take change the items in the tree to include the cluster annotations, since we are done borrowing them for the annotation step.
        let items = items
            .into_par_iter()
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
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
    C: Codec<I> + Send + Sync,
{
    /// Parallel version of [`Self::compression_cost`]
    fn par_compression_cost(&self, reference: usize, targets: &[usize]) -> usize {
        let reference = self.items[reference]
            .1
            .original()
            .unwrap_or_else(|| unreachable!("The reference item should be decompressed when computing compression costs"));
        targets
            .par_iter()
            .map(|&i| match &self.tree.items[i].1 {
                MaybeCompressed::Original(target) => I::compressed_size(&self.codec.compress(reference, target)),
                MaybeCompressed::Compressed(compressed) => I::compressed_size(compressed),
            })
            .sum()
    }

    /// Parallel version of [`Self::decompress_subtree`]
    pub(crate) fn par_decompress_subtree(&mut self, id: usize) {
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
            for (i, item) in self.par_decompressed_items(id, &targets) {
                self.tree.items[i].1 = MaybeCompressed::Original(item);
            }
        }
    }

    /// Parallel version of [`Self::decompress_child_centers`]
    pub(crate) fn par_decompress_child_centers(&mut self, id: usize) -> Option<Vec<usize>> {
        if let Some(targets) = self.get_cluster_unchecked(id).child_center_indices().map(<[_]>::to_vec) {
            for (i, item) in self.par_decompressed_items(id, &targets) {
                self.tree.items[i].1 = MaybeCompressed::Original(item);
            }
            Some(targets)
        } else {
            None
        }
    }

    /// Parallel version of [`Self::compressed_items`]
    pub(crate) fn par_compressed_items(&self, reference: usize, targets: &[usize]) -> Vec<(usize, I::Compressed)> {
        self.items[reference].1.original().map_or_else(
            // If the center is compressed, then it is impossible to have decompressed any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are decompressed and some that are compressed, so we only compress the
            // decompressed targets.
            |reference| {
                targets
                    .par_iter()
                    .filter_map(|&i| self.tree.items[i].1.original().map(|target| (i, self.codec.compress(reference, target))))
                    .collect()
            },
        )
    }

    /// Parallel version of [`Self::decompressed_items`]
    pub(crate) fn par_decompressed_items(&self, reference: usize, targets: &[usize]) -> Vec<(usize, I)> {
        self.items[reference].1.original().map_or_else(
            // If the center is compressed, we cannot decompress any of its targets, so we return an empty vector.
            Vec::new,
            // If the center is decompressed, then we may have some targets that are compressed and some that are decompressed, so we only decompress the
            // compressed targets.
            |reference| {
                targets
                    .par_iter()
                    .filter_map(|&i| {
                        self.tree.items[i]
                            .1
                            .compressed()
                            .map(|compressed| (i, self.codec.decompress(reference, compressed)))
                    })
                    .collect()
            },
        )
    }
}

impl<Id, I, T, A, M, C> PancakesTree<Id, I, T, (A, CompressionCost), M, C>
where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
    C: Codec<I> + Send + Sync,
{
    /// Parallel version of [`Self::annotate_recursive_compression_costs`]
    pub(crate) fn par_annotate_recursive_compression_costs(&mut self) {
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
            frontier
                .into_par_iter()
                .map(|id| {
                    let c = self.get_cluster_unchecked(id);
                    let cost = c.child_center_indices().map_or(c.annotation.1.unitary_cost, |child_center_indices| {
                        // If the cluster is a parent, then the cost of recursive compression is the cost of compressing the child centers in terms of the center,
                        // plus the cost of compressing the children themselves.
                        let centers_cost = self.par_compression_cost(id, child_center_indices);
                        let child_costs = child_center_indices
                            .iter()
                            .filter_map(|&cid| self.tree.items[cid].2.as_cluster().map(|c| c.annotation.1.smaller_cost()))
                            .sum::<usize>();
                        centers_cost + child_costs
                    });
                    (id, cost)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|(id, cost)| {
                    let c = self.get_cluster_unchecked_mut(id);
                    c.annotation.1.recursive_cost = Some(cost);

                    // Update the count of remaining children for the parent cluster.
                    if let Some(pid) = c.parent_center_index
                        && let Some(count) = parents_in_waiting.get_mut(&pid)
                    {
                        *count -= 1;
                    }
                });

            // The next frontier contains the parents whose children have all been processed.
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|&(_, count)| count == 0);
            frontier = full_parents.into_keys().collect();
        }
    }
}
