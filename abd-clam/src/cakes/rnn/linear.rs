//! Linear search for ranged nearest neighbors.

use distances::Number;

use crate::Dataset;

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
pub fn search<T, U, D>(data: &D, query: T, radius: U, indices: &[usize]) -> Vec<(usize, U)>
where
    T: Send + Sync + Copy,
    U: Number,
    D: Dataset<T, U>,
{
    let distances = data.query_to_many(query, indices);
    indices
        .iter()
        .copied()
        .zip(distances.into_iter())
        .filter(|&(_, d)| d <= radius)
        .collect()
}
