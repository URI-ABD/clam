//! Linear KNN search in a compressed space.

use crate::{pancakes::CodecData, Cluster, Instance};
use distances::number::UInt;

use crate::cakes::knn::Hits;

/// Perform a linear search in a compressed space.
pub fn search<I, U, M>(query: &I, k: usize, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
where
    I: Instance,
    U: UInt,
    M: Instance,
{
    let leaves = data.root().compressible_leaves();

    let mut hits = Hits::new(k);

    for leaf in leaves {
        let points = data
            .load_leaf_data(leaf)
            .unwrap_or_else(|e| unreachable!("Impossible by construction.: {e}"));
        points.into_iter().zip(leaf.indices()).for_each(|(point, index)| {
            let distance = data.metric()(query, &point);
            hits.push(index, distance);
        });
    }

    hits.extract()
}
