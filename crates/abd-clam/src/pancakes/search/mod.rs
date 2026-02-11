//! Compressive search algorithms.

use crate::{
    DistanceValue, Tree,
    cakes::{Cakes, ParSearch, RnnChess, Search},
};

use super::{Codec, MaybeCompressed};

mod approximate;
mod exact;

pub use exact::{leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

/// Nearest Neighbor Search in compressed space.
pub trait CompressiveSearch<Id, I, T, A, M>: Search<Id, I, T, A, M>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Same as [`Search::search`] but operates on a compressed tree and will decompress items as needed.
    ///
    /// # Errors
    ///
    /// - If the root center has been compressed.
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String>;
}

/// Parallel version of [`CompressiveSearch`].
pub trait ParCompressiveSearch<Id, I, T, A, M>: CompressiveSearch<Id, I, T, A, M> + ParSearch<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    /// Parallel version of [`CompressiveSearch::compressive_search`].
    ///
    /// # Errors
    ///
    /// See [`CompressiveSearch::compressive_search`] for error conditions.
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String>;
}

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for Cakes<T>
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
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

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for Cakes<T>
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
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
impl<Id, I, T, A, M, Alg> CompressiveSearch<Id, I, T, A, M> for &Alg
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    Alg: CompressiveSearch<Id, I, T, A, M>,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).compressive_search(tree, query)
    }
}

// Blanket implementations of `ParSearch` for references.
impl<Id, I, T, A, M, Alg> ParCompressiveSearch<Id, I, T, A, M> for &Alg
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Alg: ParCompressiveSearch<Id, I, T, A, M>,
{
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        (**self).par_compressive_search(tree, query)
    }
}
