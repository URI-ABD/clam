//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{Dataset, DistanceValue, Tree, utils::SizedHeap};

use super::super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnLinear(pub usize);

impl<D, M, T, A> Search<D, M, T, A> for KnnLinear
where
    D: Dataset,
    M: Fn(&D::Item, &D::Item) -> T,
    T: DistanceValue,
{
    fn name(&self) -> String {
        format!("KnnLinear(k={})", self.0)
    }

    fn search(&self, tree: &Tree<D, M, T, A>, query: &D::Item) -> Vec<(usize, T)> {
        let distances = tree.dataset.as_slice().iter().enumerate().map(|(i, (_, item))| (i, (tree.metric)(query, item)));
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances);
        heap.take_items().collect()
    }
}

impl<D, M, T, A> ParSearch<D, M, T, A> for KnnLinear
where
    D: Dataset + Send + Sync,
    D::Id: Send + Sync,
    D::Item: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&D::Item, &D::Item) -> T + Send + Sync,
    A: Send + Sync,
{
    fn par_search(&self, tree: &Tree<D, M, T, A>, query: &D::Item) -> Vec<(usize, T)> {
        let distances = tree
            .dataset
            .as_slice()
            .par_iter()
            .enumerate()
            .map(|(i, (_, item))| (i, (tree.metric)(query, item)));
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances.collect::<Vec<_>>());
        heap.take_items().collect()
    }
}
