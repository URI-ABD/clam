//! Linear search for ranged nearest neighbors.

use distances::Number;

use crate::{Dataset, Instance};

/// Linear search for the ranged nearest neighbors of a query.
///
/// # Arguments
///
/// * `data` - The dataset to search.
/// * `query` - The query to search around.
/// * `radius` - The radius to search within.
/// * `indices` - The indices to search.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<I, U, D>(data: &D, query: &I, radius: U, indices: &[usize]) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
{
    let distances = data.query_to_many(query, indices);
    indices
        .iter()
        .copied()
        .zip(distances)
        .filter(|&(_, d)| d <= radius)
        .collect()
}
