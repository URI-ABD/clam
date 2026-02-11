//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::KnnLinear,
    pancakes::{Codec, MaybeCompressed},
    utils::SizedHeap,
};

use super::super::{CompressiveSearch, ParCompressiveSearch};

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for KnnLinear
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        tree.decompress_subtree(0)?;
        let distances = tree
            .items
            .iter()
            .enumerate()
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()?;
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances);
        Ok(heap.take_items().collect())
    }
}

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for KnnLinear
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        tree.par_decompress_subtree(0)?;
        let distances = tree
            .items
            .par_iter()
            .enumerate()
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()?;
        let mut heap = SizedHeap::new(Some(self.0));
        heap.extend(distances);
        Ok(heap.take_items().collect())
    }
}
