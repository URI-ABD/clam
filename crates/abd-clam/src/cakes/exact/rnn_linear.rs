//! Ranged Nearest Neighbor (RNN) search with a naive linear scan.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{ParSearch, Search},
};

/// Ranged Nearest Neighbor (RNN) search with a naive linear scan.
///
/// The field is the radius of the query ball to search within.
pub struct RnnLinear<T: DistanceValue>(pub T);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnLinear<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("RnnLinear(radius={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        tree.distances_to_items_in_cluster(query, tree.root())
            .filter_map(|(idx, dist)| if dist <= self.0 { Some((idx, dist)) } else { None })
            .collect()
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for RnnLinear<T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        tree.par_distances_to_items_in_cluster(query, tree.root())
            .filter_map(|(idx, dist)| if dist <= self.0 { Some((idx, dist)) } else { None })
            .collect()
    }
}
