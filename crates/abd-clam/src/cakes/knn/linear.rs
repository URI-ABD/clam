//! Linear search for the k nearest neighbors of a query.

use distances::Number;

use crate::{Dataset, Instance};

use super::Hits;

/// Linear search for the nearest neighbors of a query.
///
/// # Arguments
///
/// * `data` - The dataset to search.
/// * `query` - The query to search around.
/// * `k` - The number of neighbors to search for.
/// * `indices` - The indices to search.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<I, U, D>(data: &D, query: &I, k: usize, indices: &[usize]) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    let distances = data.query_to_many(query, indices);

    let mut hits = Hits::new(k);
    indices
        .iter()
        .zip(distances.iter())
        .for_each(|(&i, &d)| hits.push(i, d));
    hits.extract()
}
