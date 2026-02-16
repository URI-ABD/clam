//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, cakes::RnnLinear};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree};

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for RnnLinear<T>
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)> {
        tree.decompress_subtree(0);
        let distances = tree
            .items
            .iter()
            .enumerate()
            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()
            .unwrap_or_else(|_| unreachable!("We just decompressed the entire tree, so there should be no errors"));
        distances.into_iter().filter(|&(_, d)| d <= self.radius).collect()
    }

    fn par_compressive_search<Query: Borrow<I> + Send + Sync>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        I::Compressed: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        C: Send + Sync,
    {
        tree.par_decompress_subtree(0);
        let distances = tree
            .items
            .par_iter()
            .enumerate()
            .map(|(i, (_, item, _))| item.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()
            .unwrap_or_else(|_| unreachable!("We just decompressed the entire tree, so there should be no errors"));
        distances.into_iter().filter(|&(_, d)| d <= self.radius).collect()
    }
}
