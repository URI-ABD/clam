//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{DistanceValue, Tree, utils::SizedHeap};

use super::super::{ParSearch, Search};

/// K-Nearest Neighbor (KNN) search with a naive linear scan.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnLinear(pub usize);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnLinear
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("KnnLinear(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let distances = tree.items.iter().enumerate().map(|(i, (_, item))| (i, (tree.metric)(query, item)));
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances);
        heap.take_items().collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnLinear
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        let distances = tree.items.par_iter().enumerate().map(|(i, (_, item))| (i, (tree.metric)(query, item)));
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances.collect::<Vec<_>>());
        heap.take_items().collect()
    }
}
