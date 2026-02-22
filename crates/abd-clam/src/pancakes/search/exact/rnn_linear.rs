//! K-Nearest Neighbor (KNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    Dataset, DistanceValue, Tree,
    cakes::RnnLinear,
    pancakes::{Codec, MaybeCompressedItem},
};

use super::super::{CompressiveSearch, ParCompressiveSearch};

impl<Item, D, M, T, A> CompressiveSearch<Item, D, M, T, A> for RnnLinear<T>
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    T: DistanceValue,
    M: Fn(&Item, &Item) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        tree.decompress_subtree(0)?;
        let distances = tree
            .dataset
            .as_slice()
            .iter()
            .enumerate()
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(distances.into_iter().filter(|&(_, d)| d <= self.0).collect())
    }
}

impl<Item, D, M, T, A> ParCompressiveSearch<Item, D, M, T, A> for RnnLinear<T>
where
    Item: Codec + Send + Sync,
    Item::Compressed: Send + Sync,
    D: Dataset<Item = MaybeCompressedItem<Item>> + Send + Sync,
    D::Id: Send + Sync,
    M: Fn(&Item, &Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        tree.par_decompress_subtree(0)?;
        let distances = tree
            .dataset
            .as_slice()
            .par_iter()
            .enumerate()
            .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
            .collect::<Result<Vec<_>, String>>()?;
        Ok(distances.into_iter().filter(|&(_, d)| d <= self.0).collect())
    }
}
