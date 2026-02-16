//! K-Nearest Neighbors (KNN) search using the Depth-First Sieve algorithm.

use core::{borrow::Borrow, cmp::Reverse};

use crate::{Cluster, DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, KnnLinear, Search, d_max, d_min};

/// K-Nearest Neighbor (KNN) search using the Depth-First Sieve algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnDfs {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
}

impl KnnDfs {
    /// Creates a new `KnnDfs` object with the specified `k`.
    pub const fn new(k: usize) -> Self {
        Self { k }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }
}

impl_named_algorithm_for_exact_knn!(KnnDfs, "knn-dfs", r"^knn-dfs::k=(\d+)$");

impl<T: DistanceValue> From<KnnDfs> for Cakes<T> {
    fn from(alg: KnnDfs) -> Self {
        Self::KnnDfs(alg)
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().search(tree, query);
        }

        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        {
            profi::prof!("KnnDfs::initialization");
            let (radius, d) = {
                let (_, item, loc) = &tree.items[0];
                let radius = loc.as_cluster().map_or_else(|| unreachable!("Root cluster not found"), |c| c.radius);
                let d = (tree.metric)(query.borrow(), item.borrow());
                (radius, d)
            };
            hits.push((0, d));
            candidates.push((0, Reverse((d_min(radius, d), d_max(radius, d), d))));
        }

        while let Some((id, Reverse((min_c, _, _)))) = candidates.pop() {
            profi::prof!("KnnDfs::while_loop");
            if hits.is_full() && hits.peek().is_some_and(|(_, &max_h)| max_h < min_c) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }

            if let Some(child_center_indices) = tree.items[id].2.as_cluster().and_then(Cluster::child_center_indices) {
                profi::prof!("KnnDfs::process_parent");
                for (child, d) in child_center_indices.iter().filter_map(|&ci| {
                    let (_, item, loc) = &tree.items[ci];
                    loc.as_cluster().map(|child| (child, (tree.metric)(query.borrow(), item.borrow())))
                }) {
                    hits.push((child.center_index, d));
                    candidates.push((child.center_index, Reverse((d_min(child.radius, d), d_max(child.radius, d), d))));
                }
            } else {
                profi::prof!("KnnDfs::process_leaf");
                let indices = tree.items[id]
                    .2
                    .as_cluster()
                    .map_or_else(|| unreachable!("Leaf cluster not found"), Cluster::subtree_range);
                hits.extend(
                    tree.items[indices]
                        .iter()
                        .enumerate()
                        .map(|(i, (_, item, _))| (i + id, (tree.metric)(query.borrow(), item.borrow()))),
                );
            }
        }

        hits.take_items().collect()
    }

    fn par_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().par_search(tree, query);
        }

        self.search(tree, query)
    }
}
