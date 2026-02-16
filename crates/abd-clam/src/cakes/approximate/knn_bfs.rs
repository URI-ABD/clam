//! Approximate version of the [`KnnBfs`](crate::cakes::KnnBfs) algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{
    Cakes, KnnLinear, Search,
    knn_bfs::{BfsCandidate, filter_candidates},
};

/// Approximate version of the [`KnnBfs`](crate::cakes::KnnBfs) algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnBfs {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
    /// This is a value between 0 and 1, where 0 implies perfect recall (i.e., the search is exact) and values closer to 1 imply stopping the search earlier,
    /// potentially sacrificing recall for faster search time.
    pub(crate) tol: f64,
}

impl_named_algorithm_for_approx_knn!(KnnBfs, "approx-knn-bfs", r"^approx-knn-bfs::k=(\d+),tol=([0-9]*\.?[0-9]+)$");

impl<T: DistanceValue> From<KnnBfs> for Cakes<T> {
    fn from(alg: KnnBfs) -> Self {
        Self::ApproxKnnBfs(alg)
    }
}

impl KnnBfs {
    /// Creates a new `KnnBfs` search object with the specified parameters.
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
    pub(crate) fn should_stop<T: DistanceValue>(&self, max_h: T, candidates: &[BfsCandidate<T>]) -> bool {
        if max_h == T::zero() {
            // If the farthest hit is at distance 0, we cannot improve any of the hits, so we stop the search.
            true
        } else {
            let max_h = max_h.to_f64().unwrap_or_else(|| unreachable!("DistanceValue should be convertible to f64"));
            let max_d_min = candidates
                .iter()
                .map(|c| crate::utils::MaxItem((), c.d_min))
                .max()
                .unwrap_or_else(|| unreachable!("candidates is non-empty"))
                .1
                .to_f64()
                .unwrap_or_else(|| unreachable!("DistanceValue should be convertible to f64"));

            let ratio = max_d_min / max_h;
            let threshold = 1.5 - ratio.clamp(0.5, 1.5);

            threshold < self.tol
        }
    }
}

impl<Id, I, T: DistanceValue, A, M: Fn(&I, &I) -> T> Search<Id, I, T, A, M> for KnnBfs {
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

        let mut candidates = Vec::new();
        let mut hits = SizedHeap::<usize, T>::new(Some(self.k));

        let root = tree.root();
        let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
        hits.push((0, d));
        candidates.push(BfsCandidate::from_cluster(root, d));

        while !candidates.is_empty() {
            let mut next_candidates = Vec::new();
            filter_candidates(&mut candidates, self.k, hits.len());

            let max_h = hits.peek().map_or_else(T::max_value, |(_, &d)| d);
            if hits.is_full() && self.should_stop(max_h, &candidates) {
                // If the stopping condition is met, we stop the search.
                break;
            }

            for candidate in candidates {
                if candidate.is_leaf || (next_candidates.len() <= self.k && (candidate.cardinality < (self.k - next_candidates.len())))
                // We still need more points to satisfy k AND The candidate cannot provide enough points to get to k
                {
                    profi::prof!("KnnBfs::process_leaf");
                    // The cluster is a leaf, so we have to look at its points
                    if candidate.is_singleton {
                        // It's a singleton, so just add non-center items with the precomputed distance
                        hits.extend(candidate.subtree_indices().map(|i| (i, d)));
                    } else {
                        // Not a singleton, so compute distances to all non-center items and add them to hits
                        let distances = candidate
                            .subtree_indices()
                            .zip(tree.items[candidate.subtree_indices()].iter())
                            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())));
                        hits.extend(distances);
                    }
                } else {
                    profi::prof!("KnnBfs::process_parent");
                    if let Some((cids, _)) = tree.items[candidate.id].2.as_cluster().and_then(|c| c.children.as_ref()) {
                        for &cid in cids {
                            let (_, item, loc) = &tree.items[cid];
                            if let Some(child) = loc.as_cluster() {
                                let d = (tree.metric)(query.borrow(), item.borrow());
                                hits.push((cid, d));
                                next_candidates.push(BfsCandidate::from_cluster(child, d));
                            }
                        }
                    }
                }
            }

            candidates = next_candidates;
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
