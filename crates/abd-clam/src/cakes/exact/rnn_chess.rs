//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

use super::super::{Cakes, RnnLinear, Search};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub struct RnnChess<T: DistanceValue> {
    /// The radius of the query ball to search within.
    pub(crate) radius: T,
}

impl<T: DistanceValue> RnnChess<T> {
    /// Creates a new `RnnChess` with the given radius.
    pub const fn new(radius: T) -> Self {
        Self { radius }
    }

    /// Creates a new `RnnLinear` with the same radius as `self`.
    pub const fn linear_variant(self) -> RnnLinear<T> {
        RnnLinear::new(self.radius)
    }
}

impl_named_algorithm_for_exact_rnn!(RnnChess, "rnn-chess", r"^rnn-chess::radius=([0-9]+(?:\.[0-9]+)?)$");

impl<T: DistanceValue> From<RnnChess<T>> for Cakes<T> {
    fn from(algorithm: RnnChess<T>) -> Self {
        Self::RnnChess(algorithm)
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        profi::prof!("RnnChess::search");

        let root = tree.root();
        let mut frontier = vec![(0, root.cardinality, root.radius, (tree.metric)(query.borrow(), tree.items[0].1.borrow()))];

        let mut hits = Vec::new();
        let mut next_frontier = Vec::new();
        let mut steps = 0;
        while !frontier.is_empty() {
            steps += 1;
            let is_sorted = frontier.windows(2).all(|w| w[0].0 <= w[1].0);
            assert!(is_sorted, "Step {steps}: Frontier size = {}, is_sorted: {is_sorted}", frontier.len());

            for (id, cardinality, radius, d) in frontier {
                profi::prof!("RnnChess::frontier_loop");

                if d <= self.radius {
                    // Center is within query radius
                    hits.push((id, d));
                }
                if d + radius <= self.radius {
                    profi::prof!("RnnChess::subsumed");

                    // Fully subsumed cluster, so we can add all items in this subtree
                    hits.extend(
                        tree.items[(id + 1)..(id + cardinality)]
                            .iter()
                            .enumerate()
                            .map(|(i, (_, item, _))| (i + id, (tree.metric)(query.borrow(), item.borrow()))),
                    );
                } else if let Some(child_center_ids) = tree.items[id].2.as_cluster().and_then(|c| c.child_center_indices()) {
                    profi::prof!("RnnChess::parent");

                    // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                    next_frontier.extend(child_center_ids.iter().filter_map(|&cid| {
                        let (_, item, loc) = &tree.items[cid];
                        loc.as_cluster().and_then(|child| {
                            let dist = (tree.metric)(query.borrow(), item.borrow());
                            if dist <= self.radius + child.radius {
                                Some((cid, child.cardinality, child.radius, dist))
                            } else {
                                None
                            }
                        })
                    }));
                } else {
                    profi::prof!("RnnChess::overlapping");

                    // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                    hits.extend(tree.items[(id + 1)..(id + cardinality)].iter().enumerate().filter_map(|(i, (_, item, _))| {
                        let dist = (tree.metric)(query.borrow(), item.borrow());
                        if dist <= self.radius { Some((i + id, dist)) } else { None }
                    }));
                }
            }
            frontier = core::mem::take(&mut next_frontier);
        }

        hits
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
        profi::prof!("RnnChess::par_search");

        let root = tree.root();
        let mut frontier = vec![(0, root.cardinality, root.radius, (tree.metric)(query.borrow(), tree.items[0].1.borrow()))];

        let mut hits = Vec::new();
        let mut next_frontier = Vec::new();
        while !frontier.is_empty() {
            for (id, cardinality, radius, d) in frontier {
                profi::prof!("RnnChess::par_frontier_loop");

                if d <= self.radius {
                    // Center is within query radius
                    hits.push((id, d));
                }
                if d + radius <= self.radius {
                    profi::prof!("RnnChess::par_subsumed");

                    // Fully subsumed cluster, so we can add all items in this subtree
                    hits.par_extend(
                        tree.items[(id + 1)..(id + cardinality)]
                            .par_iter()
                            .enumerate()
                            .map(|(i, (_, item, _))| (i + id, (tree.metric)(query.borrow(), item.borrow()))),
                    );
                } else if let Some(child_center_ids) = tree.items[id].2.as_cluster().and_then(|c| c.child_center_indices()) {
                    profi::prof!("RnnChess::par_parent");

                    // Parent cluster is partially overlapping, so we need to check children for overlap and add them to the frontier
                    next_frontier.par_extend(child_center_ids.par_iter().filter_map(|&cid| {
                        let (_, item, loc) = &tree.items[cid];
                        loc.as_cluster().and_then(|child| {
                            let dist = (tree.metric)(query.borrow(), item.borrow());
                            if dist <= self.radius + child.radius {
                                Some((cid, child.cardinality, child.radius, dist))
                            } else {
                                None
                            }
                        })
                    }));
                } else {
                    profi::prof!("RnnChess::par_overlapping");

                    // Leaf cluster and not fully subsumed, so we need to check all items in this cluster
                    hits.par_extend(tree.items[(id + 1)..(id + cardinality)].par_iter().enumerate().filter_map(|(i, (_, item, _))| {
                        let dist = (tree.metric)(query.borrow(), item.borrow());
                        if dist <= self.radius { Some((i + id, dist)) } else { None }
                    }));
                }
            }
            frontier = core::mem::take(&mut next_frontier);
        }

        hits
    }
}
