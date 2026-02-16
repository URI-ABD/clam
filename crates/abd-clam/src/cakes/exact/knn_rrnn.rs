//! K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree, utils::MinItem};

use super::super::{Cakes, KnnLinear, RnnChess, Search, d_max};

/// K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnRrnn {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
}

impl KnnRrnn {
    /// Creates a new `KnnRrnn` object with the given `k`.
    pub const fn new(k: usize) -> Self {
        Self { k }
    }

    /// Returns a `KnnLinear` object with the same `k`.
    pub const fn linear_variant(self) -> KnnLinear {
        KnnLinear { k: self.k }
    }
}

impl_named_algorithm_for_exact_knn!(KnnRrnn, "knn-rrnn", r"^knn-rrnn::k=(\d+)$");

impl<T: DistanceValue> From<KnnRrnn> for Cakes<T> {
    fn from(algorithm: KnnRrnn) -> Self {
        Self::KnnRrnn(algorithm)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnRrnn
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().search(tree, query);
        }

        let mut latest = tree.root();
        let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
        let mut candidate_radii = Vec::with_capacity(100);
        candidate_radii.extend([MinItem((), d), MinItem((), d_max(latest.radius, d))]);
        while let Some(MinItem(child, d)) = latest.child_center_indices().and_then(|cids| {
            cids.iter()
                .filter_map(|&cid| {
                    let (_, item, loc) = &tree.items[cid];
                    loc.as_cluster().map(|child| MinItem(child, (tree.metric)(query.borrow(), item.borrow())))
                })
                .min()
        }) {
            latest = child;
            candidate_radii.extend([MinItem((), d), MinItem((), d_max(latest.radius, d))]);
        }
        candidate_radii.sort();

        let mut hits = Vec::with_capacity(self.k * 2);
        for MinItem((), d) in candidate_radii {
            hits = RnnChess::new(d).search(tree, query);
            if hits.len() >= self.k {
                hits.sort_by_key(|&(_, d)| MinItem((), d));
                hits.truncate(self.k);
                break;
            }
        }
        hits
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

        let mut latest = tree.root();
        let d = (tree.metric)(query.borrow(), tree.items[0].1.borrow());
        let mut candidate_radii = Vec::with_capacity(100);
        candidate_radii.extend([MinItem((), d), MinItem((), d_max(latest.radius, d))]);
        while let Some(MinItem(child, d)) = latest.child_center_indices().and_then(|cids| {
            cids.par_iter()
                .filter_map(|&cid| {
                    let (_, item, loc) = &tree.items[cid];
                    loc.as_cluster().map(|child| MinItem(child, (tree.metric)(query.borrow(), item.borrow())))
                })
                .min()
        }) {
            latest = child;
            candidate_radii.extend([MinItem((), d), MinItem((), d_max(latest.radius, d))]);
        }
        candidate_radii.sort();

        let mut hits = Vec::with_capacity(self.k * 2);
        for MinItem((), d) in candidate_radii {
            hits = RnnChess::new(d).par_search(tree, query);
            if hits.len() >= self.k {
                hits.sort_by_key(|&(_, d)| MinItem((), d));
                hits.truncate(self.k);
                break;
            }
        }
        hits
    }
}
