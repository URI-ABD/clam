//! Compressive search algorithms.

use crate::{
    Dataset, DistanceValue, Tree,
    cakes::{Cakes, RnnChess},
};

use super::{Codec, MaybeCompressedItem};

mod approximate;
mod exact;

pub use exact::{leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

/// Nearest Neighbor Search in compressed space.
pub trait CompressiveSearch<Item, D, M, T, A>
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    M: Fn(&Item, &Item) -> T,
    T: DistanceValue,
{
    /// Same as [`Search::search`] but operates on a compressed tree and will decompress items as needed.
    ///
    /// # Errors
    ///
    /// - If the root center has been compressed.
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String>;
}

/// Parallel version of [`CompressiveSearch`].
pub trait ParCompressiveSearch<Item, D, M, T, A>: CompressiveSearch<Item, D, M, T, A>
where
    Item: Codec + Send + Sync,
    Item::Compressed: Send + Sync,
    D: Dataset<Item = MaybeCompressedItem<Item>> + Send + Sync,
    D::Id: Send + Sync,
    M: Fn(&Item, &Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`CompressiveSearch::compressive_search`].
    ///
    /// # Errors
    ///
    /// See [`CompressiveSearch::compressive_search`] for error conditions.
    fn par_compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String>;
}

impl<Item, D, M, T, A> CompressiveSearch<Item, D, M, T, A> for Cakes<T>
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    T: DistanceValue,
    M: Fn(&Item, &Item) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        match self {
            Self::KnnBfs(alg) => alg.compressive_search(tree, query),
            Self::KnnDfs(alg) => alg.compressive_search(tree, query),
            Self::KnnLinear(alg) => alg.compressive_search(tree, query),
            Self::KnnRrnn(alg) => alg.compressive_search(tree, query),
            Self::RnnChess(alg) => alg.compressive_search(tree, query),
            Self::RnnLinear(alg) => alg.compressive_search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.compressive_search(tree, query),
        }
    }
}

impl<Item, D, M, T, A> ParCompressiveSearch<Item, D, M, T, A> for Cakes<T>
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
        match self {
            Self::KnnBfs(alg) => alg.par_compressive_search(tree, query),
            Self::KnnDfs(alg) => alg.par_compressive_search(tree, query),
            Self::KnnLinear(alg) => alg.par_compressive_search(tree, query),
            Self::KnnRrnn(alg) => alg.par_compressive_search(tree, query),
            Self::RnnChess(alg) => alg.par_compressive_search(tree, query),
            Self::RnnLinear(alg) => alg.par_compressive_search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.par_compressive_search(tree, query),
        }
    }
}

// Blanket implementations of `Search` for references.
impl<Item, D, M, T, A, Alg> CompressiveSearch<Item, D, M, T, A> for &Alg
where
    Item: Codec,
    D: Dataset<Item = MaybeCompressedItem<Item>>,
    M: Fn(&Item, &Item) -> T,
    T: DistanceValue,
    Alg: CompressiveSearch<Item, D, M, T, A>,
{
    fn compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        (**self).compressive_search(tree, query)
    }
}

// Blanket implementations of `ParSearch` for references.
impl<Item, D, M, T, A, Alg> ParCompressiveSearch<Item, D, M, T, A> for &Alg
where
    Item: Codec + Send + Sync,
    Item::Compressed: Send + Sync,
    D: Dataset<Item = MaybeCompressedItem<Item>> + Send + Sync,
    D::Id: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&Item, &Item) -> T + Send + Sync,
    Alg: ParCompressiveSearch<Item, D, M, T, A>,
{
    fn par_compressive_search(&self, tree: &mut Tree<D, M, T, A>, query: &Item) -> Result<Vec<(usize, T)>, String> {
        (**self).par_compressive_search(tree, query)
    }
}
