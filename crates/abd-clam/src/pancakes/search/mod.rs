//! Compressive search algorithms.

use core::borrow::Borrow;

use crate::{
    DistanceValue,
    cakes::{Cakes, Search},
};

use super::{Codec, Compressible, PancakesTree};

mod approximate;
mod exact;

pub use exact::{leaf_into_hits, par_leaf_into_hits, par_pop_till_leaf, pop_till_leaf};

/// An extension of [`Search`] that can perform search on a compressed tree and will only decompress items as needed to compute their distances to the query.
pub trait CompressiveSearch<Id, I, T, A, M, C>: Search<Id, I, T, A, M>
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    /// Same as [`Search::search`] but operates on a compressed tree and will decompress items as needed.
    ///
    /// This requires a mutable reference to the tree, since we may need to decompress items in the tree as we search. For the algorithms we provide in this
    /// crate, the returned hits are guaranteed to be the same as the hits returned by [`Search::search`] on the fully decompressed tree, and those specific
    /// items will have been decompressed by the time this method returns.
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>;

    /// Parallel version of [`CompressiveSearch::compressive_search`].
    fn par_compressive_search<Query: Borrow<I> + Send + Sync>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        I::Compressed: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        C: Send + Sync;
}

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for Cakes<T>
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)> {
        match self {
            Self::KnnBfs(alg) => alg.compressive_search(tree, query),
            Self::KnnDfs(alg) => alg.compressive_search(tree, query),
            Self::KnnLinear(alg) => alg.compressive_search(tree, query),
            Self::KnnRrnn(alg) => alg.compressive_search(tree, query),
            Self::KnnSieve(_) => todo!("Compressive search for KnnSieve is not yet implemented"),
            Self::RnnChess(alg) => alg.compressive_search(tree, query),
            Self::RnnLinear(alg) => alg.compressive_search(tree, query),
            Self::ApproxKnnBfs(_) => todo!("Compressive search for ApproxKnnBfs is not yet implemented"),
            Self::ApproxKnnDfs(alg) => alg.compressive_search(tree, query),
            Self::ApproxKnnSieve(_) => todo!("Compressive search for ApproxKnnSieve is not yet implemented"),
        }
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
        match self {
            Self::KnnBfs(alg) => alg.par_compressive_search(tree, query),
            Self::KnnDfs(alg) => alg.par_compressive_search(tree, query),
            Self::KnnLinear(alg) => alg.par_compressive_search(tree, query),
            Self::KnnRrnn(alg) => alg.par_compressive_search(tree, query),
            Self::KnnSieve(_) => todo!("Compressive search for KnnSieve is not yet implemented"),
            Self::RnnChess(alg) => alg.par_compressive_search(tree, query),
            Self::RnnLinear(alg) => alg.par_compressive_search(tree, query),
            Self::ApproxKnnBfs(_) => todo!("Compressive search for ApproxKnnBfs is not yet implemented"),
            Self::ApproxKnnDfs(alg) => alg.par_compressive_search(tree, query),
            Self::ApproxKnnSieve(_) => todo!("Compressive search for ApproxKnnSieve is not yet implemented"),
        }
    }
}
