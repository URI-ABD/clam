//! Clustered Ranged Nearest Neighbor search in a compressed space.

use distances::number::UInt;
use rayon::prelude::*;

use crate::{
    pancakes::{CodecData, SquishyBall},
    Cluster, Instance,
};

/// Perform a clustered search in a compressed space.
pub fn search<I, U, M>(query: &I, radius: U, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let [confirmed, straddlers] = tree_search(query, radius, data);
    leaf_search(query, radius, data, confirmed, straddlers)
}

/// Perform coarse-grained tree search.
///
/// # Arguments
///
/// * `query` - The query to search around.
/// * `radius` - The radius to search within.
/// * `data` - The data to search.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the confirmed clusters, i.e.
/// those that are contained within the query ball, and the second element is the
/// straddlers, i.e. those that overlap the query ball. The 2-tuples are the clusters
/// and the distance from the query to the cluster center.
fn tree_search<'a, I, U, M>(query: &I, radius: U, data: &'a CodecData<I, U, M>) -> [Vec<(&'a SquishyBall<U>, U)>; 2]
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let mut confirmed = Vec::new();
    let mut straddlers = Vec::new();
    let mut candidates = vec![data.root()];

    let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
    while !candidates.is_empty() {
        (terminal, non_terminal) = candidates
            .into_par_iter()
            .map(|c| {
                let center = &data.centers()[&c.arg_center()];
                let distance = data.metric()(center, query);
                (c, distance)
            })
            .filter(|&(c, d)| d <= (c.radius() + radius))
            .partition(|&(c, d)| ((c.radius() + d) <= radius));
        confirmed.append(&mut terminal);

        (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.squish());
        straddlers.append(&mut terminal);

        candidates = non_terminal
            .into_iter()
            .flat_map(|(c, _)| {
                // if d < c.radius() {
                //     c.overlapping_children(data, query, radius)
                // } else {
                //     c.children().map_or_else(|| unreachable!("Non-leaf node without children"), |v| v.to_vec())
                // }
                c.children()
                    .map_or_else(|| unreachable!("Non-leaf node without children"), |v| v.to_vec())
            })
            .collect();
    }

    [confirmed, straddlers]
}

/// Perform fine-grained leaf search.
fn leaf_search<I, U, M>(
    query: &I,
    radius: U,
    data: &CodecData<I, U, M>,
    confirmed: Vec<(&SquishyBall<U>, U)>,
    straddlers: Vec<(&SquishyBall<U>, U)>,
) -> Vec<(usize, U)>
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    // TODO: Check this monstrosity

    let hits = confirmed.into_iter().flat_map(|(c, _)| {
        c.compressible_leaves()
            .into_iter()
            .flat_map(|leaf| {
                let points = data
                    .load_leaf_data(leaf)
                    .unwrap_or_else(|e| unreachable!("Leaf data not found: {e}"));
                points.into_iter().map(|p| data.metric()(query, &p)).zip(leaf.indices())
            })
            .filter(|&(d, _)| d <= radius)
            .map(|(d, i)| (i, d))
    });

    straddlers
        .into_iter()
        .flat_map(|(c, _)| {
            c.compressible_leaves()
                .into_iter()
                .flat_map(|leaf| {
                    let points = data
                        .load_leaf_data(leaf)
                        .unwrap_or_else(|e| unreachable!("Leaf data not found: {e}"));
                    points.into_iter().map(|p| data.metric()(query, &p)).zip(leaf.indices())
                })
                .filter(|(d, _)| *d <= radius)
                .map(|(d, i)| (i, d))
        })
        .chain(hits)
        .collect()
}
