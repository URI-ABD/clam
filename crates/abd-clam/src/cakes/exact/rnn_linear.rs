//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{Dataset, DistanceValue, Tree};

use super::super::{ParSearch, Search};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
///
/// The field is the radius of the query ball to search within.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<D, M, T, A> Search<D, M, T, A> for RnnLinear<T>
where
    D: Dataset,
    M: Fn(&D::Item, &D::Item) -> T,
    T: DistanceValue,
{
    fn name(&self) -> String {
        format!("RnnLinear(radius={})", self.0)
    }

    fn search(&self, tree: &Tree<D, M, T, A>, query: &D::Item) -> Vec<(usize, T)> {
        tree.dataset
            .as_slice()
            .iter()
            .enumerate()
            .filter_map(|(i, (_, item))| {
                let d = (tree.metric)(query, item);
                if d <= self.0 { Some((i, d)) } else { None }
            })
            .collect()
    }
}

impl<D, M, T, A> ParSearch<D, M, T, A> for RnnLinear<T>
where
    D: Dataset + Send + Sync,
    D::Id: Send + Sync,
    D::Item: Send + Sync,
    M: Fn(&D::Item, &D::Item) -> T + Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
{
    fn par_search(&self, tree: &Tree<D, M, T, A>, query: &D::Item) -> Vec<(usize, T)> {
        tree.dataset
            .as_slice()
            .par_iter()
            .enumerate()
            .filter_map(|(i, (_, item))| {
                let d = (tree.metric)(query, item);
                if d <= self.0 { Some((i, d)) } else { None }
            })
            .collect()
    }
}
