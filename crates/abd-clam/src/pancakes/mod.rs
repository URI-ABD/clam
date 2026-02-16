//! Compression and compressive search algorithms.
//!
//! This module enables the [`Tree::compress`](crate::Tree::compress) to create a [`PancakesTree`]. The [`PancakesTree::decompress`] method can be used to
//! invert the compression operation, and recover a `Tree` that contains the same items as the original tree, but may have trimmed several clusters that were
//! removed during the compression step. These methods require that the items stored in the tree implement the [`Compressible`] trait using a codec that
//! implements a corresponding compression/decompression scheme using the [`Codec`] trait.
//!
//! The compressed trees can be used with the [`CompressiveSearch`] algorithms, which are extensions of the [`Search`](crate::cakes::Search) algorithms for
//! performing search in a compressed space while only decompressing those items that are needed to compute the distances to a query.

use std::collections::HashMap;

use crate::{DistanceValue, Tree};

mod codec;
mod search;
mod tree;

pub use codec::{Codec, Compressible, MaybeCompressed};
pub use search::CompressiveSearch;

/// A tree with items of type `I` that can be compressed using a codec of type `C`.
///
/// Use the [`Tree::compress`] or [`Tree::par_compress`] method to create a `PancakesTree` from a `Tree`.
pub struct PancakesTree<Id, I, T, A, M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    /// The underlying tree.
    tree: Tree<Id, MaybeCompressed<I>, T, A, M>,
    /// The codec used for compression and decompression.
    codec: C,
}

impl<Id, I, T, A, M, C> core::ops::Deref for PancakesTree<Id, I, T, A, M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    type Target = Tree<Id, MaybeCompressed<I>, T, A, M>;

    fn deref(&self) -> &Self::Target {
        &self.tree
    }
}

impl<Id, I, T, A, M, C> core::ops::DerefMut for PancakesTree<Id, I, T, A, M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tree
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    I: Compressible,
    T: DistanceValue,
{
    /// Compresses the tree.
    ///
    /// The compression is done using the following steps:
    ///
    /// 1. Annotate the clusters with their unitary compression costs, i.e. the total cost of compressing all non-center items in the cluster in terms of the
    ///    center.
    /// 2. Annotate the clusters with their recursive compression costs, i.e. the total cost of compressing the child centers of the cluster and the smaller of
    ///    the unitary and recursive compression costs of the child clusters, using a bottom-up traversal of the tree.
    /// 3. Trim the tree to only retain the first cluster along each branch for which unitary compression is cheaper than recursive compression, and remove the
    ///    descendants of those clusters, using a top-down traversal of the tree. The parameter `min_depth` can be used to specify a minimum depth for the
    ///    clusters to retain.
    /// 4. Compress the items in the tree from the root down to the new leaves, using a top-down traversal of the tree. Each parent cluster uses recursive
    ///    compression, and each leaf cluster uses unitary compression.
    ///
    /// Note that the center of the root cluster is never compressed, so the tree always contains at least one uncompressed item.
    pub fn compress<C: Codec<I>>(self, codec: C, min_depth: usize) -> PancakesTree<Id, I, T, A, M, C> {
        let mut tree = self.annotate_unitary_compression_costs(codec);
        tree.annotate_recursive_compression_costs();
        tree.trim_to_unitary_clusters(min_depth);
        tree.compress_from_root();
        let PancakesTree { tree, codec } = tree;
        let tree = tree.decompound_annotations().0;
        PancakesTree { tree, codec }
    }
}

impl<Id, I, T, A, M, C> PancakesTree<Id, I, T, A, M, C>
where
    I: Compressible,
    C: Codec<I>,
{
    /// Gets a reference to the codec used for compression and decompression.
    pub const fn codec(&self) -> &C {
        &self.codec
    }

    /// Takes ownership of the underlying tree and the codec.
    pub fn into_inner(self) -> (Tree<Id, MaybeCompressed<I>, T, A, M>, C) {
        let Self { tree, codec } = self;
        (tree, codec)
    }

    /// Inverts the [`Tree::compress`] operation, except that it cannot recover the clusters that were removed in the trimming step.
    pub fn decompress(mut self) -> (Tree<Id, I, T, A, M>, C) {
        self.decompress_subtree(0);

        let Self { tree, codec } = self;
        let Tree { items, metric } = tree;
        let items = items
            .into_iter()
            .filter_map(|(id, item, loc)| item.take_original().map(|item| (id, item, loc)))
            .collect();

        let tree = Tree { items, metric };
        (tree, codec)
    }

    /// Compresses the tree from the root cluster to the leaves.
    ///
    /// This can be used to reset the tree to its fully compressed state after it has been partially decompressed in applications such as compressive search.
    pub fn compress_from_root(&mut self) {
        let (leaves, parents) = self.iter_clusters().partition::<Vec<_>, _>(|c| c.is_leaf());
        let mut frontier = leaves.into_iter().map(|c| c.center_index).collect::<Vec<_>>();

        let mut parents_in_waiting = parents
            .into_iter()
            .map(|c| (c.center_index, (c.child_center_indices().map_or(0, <[_]>::len))))
            .collect::<HashMap<_, _>>();
        let mut full_parents: HashMap<_, _>;

        while !frontier.is_empty() {
            for id in frontier {
                let c = self.get_cluster_unchecked(id);
                let pid = c.parent_center_index;
                // Get the targets to compress in terms of the center.
                let targets = c.child_center_indices().map_or_else(
                    || c.subtree_range().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                    <[_]>::to_vec,                  // If the cluster is a parent, we only compress the child centers.
                );
                // Compress the targets and overwrite the original items with the compressed ones.
                for (i, compressed) in self.compressed_items(c.center_index, &targets) {
                    self.tree.items[i].1 = MaybeCompressed::Compressed(compressed);
                }
                // Update the count of remaining children for the parent cluster.
                if let Some(pid) = pid
                    && let Some(count) = parents_in_waiting.get_mut(&pid)
                {
                    *count -= 1;
                }
            }

            // Update the frontier to the parents whose children have all been processed.
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|&(_, count)| count == 0);
            frontier = full_parents.into_keys().collect();
        }
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::compress`]
    pub fn par_compress<C: Codec<I> + Send + Sync>(self, codec: C, min_depth: usize) -> PancakesTree<Id, I, T, A, M, C> {
        let mut tree = self.par_annotate_unitary_compression_costs(codec);
        tree.par_annotate_recursive_compression_costs();
        tree.trim_to_unitary_clusters(min_depth);
        tree.par_compress_from_root();
        let PancakesTree { tree, codec } = tree;
        let tree = tree.decompound_annotations().0;
        PancakesTree { tree, codec }
    }
}

impl<Id, I, T, A, M, C> PancakesTree<Id, I, T, A, M, C>
where
    Id: Send + Sync,
    I: Compressible + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
    C: Codec<I> + Send + Sync,
{
    /// Parallel version of [`Self::decompress`]
    pub fn par_decompress(mut self) -> (Tree<Id, I, T, A, M>, C) {
        self.par_decompress_subtree(0);

        let Self { tree, codec } = self;
        let Tree { items, metric } = tree;
        let items = items
            .into_iter()
            .filter_map(|(id, item, loc)| item.take_original().map(|item| (id, item, loc)))
            .collect();

        let tree = Tree { items, metric };
        (tree, codec)
    }

    /// Parallel version of [`Self::compress_from_root`]
    pub fn par_compress_from_root(&mut self) {
        let (leaves, parents) = self.iter_clusters().partition::<Vec<_>, _>(|c| c.is_leaf());
        let mut frontier = leaves.into_iter().map(|c| c.center_index).collect::<Vec<_>>();

        let mut parents_in_waiting = parents
            .into_iter()
            .map(|c| (c.center_index, (c.child_center_indices().map_or(0, <[_]>::len))))
            .collect::<HashMap<_, _>>();
        let mut full_parents: HashMap<_, _>;

        while !frontier.is_empty() {
            for id in frontier {
                let c = self.get_cluster_unchecked(id);
                let pid = c.parent_center_index;
                // Get the targets to compress in terms of the center.
                let targets = c.child_center_indices().map_or_else(
                    || c.subtree_range().collect(), // If the cluster is a leaf, we compress all the non-center items in the cluster.
                    <[_]>::to_vec,                  // If the cluster is a parent, we only compress the child centers.
                );
                // Compress the targets and overwrite the original items with the compressed ones.
                for (i, compressed) in self.par_compressed_items(c.center_index, &targets) {
                    self.tree.items[i].1 = MaybeCompressed::Compressed(compressed);
                }
                // Update the count of remaining children for the parent cluster.
                if let Some(pid) = pid
                    && let Some(count) = parents_in_waiting.get_mut(&pid)
                {
                    *count -= 1;
                }
            }

            // Update the frontier to the parents whose children have all been processed.
            (full_parents, parents_in_waiting) = parents_in_waiting.into_iter().partition(|&(_, count)| count == 0);
            frontier = full_parents.into_keys().collect();
        }
    }
}
