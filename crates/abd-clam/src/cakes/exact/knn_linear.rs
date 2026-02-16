//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree, utils::SizedHeap};

use super::super::{Cakes, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct KnnLinear {
    /// The number of nearest neighbors to find.
    pub(crate) k: usize,
}

impl KnnLinear {
    /// Creates a new `KnnLinear` search object with the specified number of nearest neighbors to find.
    pub const fn new(k: usize) -> Self {
        Self { k }
    }
}

impl_named_algorithm_for_exact_knn!(KnnLinear, "knn-linear", r"^knn-linear::k=(\d+)$");

impl<T: DistanceValue> From<KnnLinear> for Cakes<T> {
    fn from(alg: KnnLinear) -> Self {
        Self::KnnLinear(alg)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnLinear
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        let distances = tree
            .items
            .iter()
            .enumerate()
            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())));
        let mut heap = SizedHeap::new(Some(self.k));
        heap.extend(distances);
        heap.take_items().collect()
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
        let distances = tree
            .items
            .par_iter()
            .enumerate()
            .map(|(i, (_, item, _))| (i, (tree.metric)(query.borrow(), item.borrow())));
        let mut heap = SizedHeap::new(Some(self.k));
        heap.extend(distances.collect::<Vec<_>>());
        heap.take_items().collect()
    }
}
