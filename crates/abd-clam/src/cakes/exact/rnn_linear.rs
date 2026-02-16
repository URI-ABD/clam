//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::super::{Cakes, Search};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
#[must_use]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct RnnLinear<T: DistanceValue> {
    /// The radius of the query ball to search within.
    pub(crate) radius: T,
}

impl<T: DistanceValue> RnnLinear<T> {
    /// Creates a new `RnnLinear` with the given radius.
    pub const fn new(radius: T) -> Self {
        Self { radius }
    }
}

impl_named_algorithm_for_exact_rnn!(RnnLinear, "rnn-linear", r"^rnn-linear::radius=([0-9]+(?:\.[0-9]+)?)$");

impl<T: DistanceValue> From<RnnLinear<T>> for Cakes<T> {
    fn from(algorithm: RnnLinear<T>) -> Self {
        Self::RnnLinear(algorithm)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnLinear<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        tree.items
            .iter()
            .enumerate()
            .filter_map(|(i, (_, item, _))| {
                let d = (tree.metric)(query.borrow(), item.borrow());
                if d <= self.radius { Some((i, d)) } else { None }
            })
            .collect()
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
        tree.items
            .par_iter()
            .enumerate()
            .filter_map(|(i, (_, item, _))| {
                let d = (tree.metric)(query.borrow(), item.borrow());
                if d <= self.radius { Some((i, d)) } else { None }
            })
            .collect()
    }
}
