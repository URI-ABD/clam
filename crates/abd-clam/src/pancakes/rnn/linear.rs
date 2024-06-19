//! Linear search in a compressed space.

use distances::number::UInt;
use rayon::prelude::*;

use crate::{pancakes::CodecData, Cluster, Instance};

/// Perform a linear search in a compressed space.
pub fn search<I, U, M>(query: &I, radius: U, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let leaves = data.root().compressible_leaves();

    leaves
        .into_par_iter()
        .flat_map(|leaf| {
            let points = data
                .load_leaf_data(leaf)
                .unwrap_or_else(|e| unreachable!("Impossible by construction.: {e}"));
            points
                .into_par_iter()
                .zip(leaf.indices().into_par_iter())
                .filter_map(|(point, index)| {
                    let distance = data.metric()(query, &point);
                    if distance <= radius {
                        Some((index, distance))
                    } else {
                        None
                    }
                })
        })
        .collect()
}
