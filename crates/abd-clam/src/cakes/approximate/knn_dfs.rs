//! Approximate version of the [`KnnDfs`](crate::cakes::KnnDfs) algorithm.

use core::{borrow::Borrow, cmp::Reverse};

use rayon::prelude::*;

use crate::{Cluster, DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, KnnLinear, Search, d_max, d_min};

/// Approximate version of the [`KnnDfs`](crate::cakes::KnnDfs) algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnDfs {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
    /// The ratio of the distance to the closest candidate to the distance to the farthest hit at which we stop the search.
    ///
    /// This is a value between 0 and 1, where 0 implies perfect recall (i.e., the search is exact) and values closer to 1 imply stopping the search earlier,
    /// potentially sacrificing recall for faster search time.
    pub(crate) tol: f64,
}

impl_named_algorithm_for_approx_knn!(KnnDfs, "approx-knn-dfs", r"^approx-knn-dfs::k=(\d+),tol=([0-9]*\.?[0-9]+)$");

impl<T: DistanceValue> From<KnnDfs> for Cakes<T> {
    fn from(alg: KnnDfs) -> Self {
        Self::ApproxKnnDfs(alg)
    }
}

impl KnnDfs {
    /// Creates a new `KnnDfs` search object with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `k` - The number of nearest neighbors to find.
    /// * `tol` - The ratio of the distance to the closest candidate to the distance to the farthest hit at which we stop the search.
    pub const fn new(k: usize, tol: f64) -> Self {
        Self { k, tol: tol.clamp(0.0, 1.0) }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }

    /// Checks whether we should stop the search.
    pub(crate) fn should_stop<T: DistanceValue>(&self, min_c: T, max_h: T) -> bool {
        if max_h == T::zero() {
            // If the farthest hit is at distance 0, we cannot improve any of the hits, so we stop the search.
            true
        } else {
            let min_c = min_c
                .to_f64()
                .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
            let max_h = max_h
                .to_f64()
                .unwrap_or_else(|| unreachable!("Distance value should always be convertible to f64"));
            let ratio = 1.0 - (min_c / max_h);
            // We continue the search if the closest candidate has the potential to improve our hits by enough.
            ratio < self.tol
        }
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnDfs {
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .iter()
                .enumerate()
                .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                .collect();
        }

        let mut candidates = SizedHeap::<usize, Reverse<(T, T, T)>>::new(None);
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        {
            profi::prof!("ApproxKnnDfs::initialization");
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
            profi::prof!("ApproxKnnDfs::while_loop");
            if hits.is_full() && hits.peek().is_some_and(|(_, &max_h)| max_h < min_c && self.should_stop(min_c, max_h)) {
                // The closest candidate cannot improve our hits, so we can stop.
                break;
            }

            if let Some(child_center_indices) = tree.items[id].2.as_cluster().and_then(Cluster::child_center_indices) {
                profi::prof!("ApproxKnnDfs::process_parent");
                for (child, d) in child_center_indices.iter().filter_map(|&ci| {
                    let (_, item, loc) = &tree.items[ci];
                    loc.as_cluster().map(|child| (child, (tree.metric)(query.borrow(), item.borrow())))
                }) {
                    hits.push((child.center_index, d));
                    candidates.push((child.center_index, Reverse((d_min(child.radius, d), d_max(child.radius, d), d))));
                }
            } else {
                profi::prof!("ApproxKnnDfs::process_leaf");
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
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())))
                .collect();
        }

        self.search(tree, query)
    }
}
