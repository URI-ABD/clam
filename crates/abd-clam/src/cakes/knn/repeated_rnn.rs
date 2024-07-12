//! Repeated RNN search, with increasing radii, for k-nearest neighbors.

use distances::Number;

use crate::{cakes::rnn::clustered, utils, Cluster, Dataset, Instance, Tree};

use super::Hits;

/// The multiplier to use for increasing the radius in the repeated RNN algorithm.
const MULTIPLIER: f64 = 2.0;

/// K-Nearest Neighbor search using a repeated RNN search.
///
/// # Arguments
///
/// * `tree` - The tree to search.
/// * `query` - The query to search around.
/// * `k` - The number of neighbors to search for.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<I, U, D, C>(tree: &Tree<I, U, D, C>, query: &I, k: usize) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let mut radius = f64::EPSILON + tree.radius().as_f64() / tree.cardinality().as_f64();
    let [mut confirmed, mut straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));

    let mut num_confirmed = count_hits(&confirmed);

    while num_confirmed == 0 {
        radius *= MULTIPLIER;
        [confirmed, straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    while num_confirmed < k {
        let lfd: f64 = utils::mean(
            &confirmed
                .iter()
                .chain(straddlers.iter())
                .map(|&(c, _)| c.lfd())
                .collect::<Vec<_>>(),
        );
        let factor = (k.as_f64() / num_confirmed.as_f64()).powf(1. / (lfd + f64::EPSILON));

        radius *= if factor < MULTIPLIER { factor } else { MULTIPLIER };
        [confirmed, straddlers] = clustered::tree_search(tree.data(), &tree.root, query, U::from(radius));
        num_confirmed = count_hits(&confirmed);
    }

    Hits::from_vec(
        k,
        clustered::leaf_search(&tree.data, confirmed, straddlers, query, U::from(radius)),
    )
    .extract()
}

/// Count the total cardinality of the clusters.
fn count_hits<U: Number, C: Cluster<U>>(clusters: &[(&C, U)]) -> usize {
    clusters.iter().map(|(c, _)| c.cardinality()).sum()
}
