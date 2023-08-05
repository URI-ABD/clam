//! Algorithms for Ranged Nearest Neighbor search.
//!
//! The stable algorithms are `Linear` and `Clustered`, with the default being `Clustered`.
//!
//! We will experiment with other algorithms in the future, and they will be added to this
//! module as they are being implemented. They should not be considered stable until they
//! are documented as such.

use distances::Number;

use crate::{Dataset, Tree};

pub(crate) mod clustered;
pub(crate) mod linear;

/// The algorithm to use for Ranged Nearest Neighbor search.
///
/// The default is `Clustered`, as determined by the benchmarks in the crate.
#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    /// Use linear search on the entire dataset.
    ///
    /// This is a stable algorithm.
    Linear,

    /// Use a clustered search, as described in the CHESS paper.
    ///
    /// This is a stable algorithm.
    Clustered,
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::Clustered
    }
}

impl Algorithm {
    /// Searches for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `radius` - The radius to search within.
    /// * `tree` - The tree to search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn search<T, U, D>(self, query: T, radius: U, tree: &Tree<T, U, D>) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        match self {
            Self::Linear => linear::search(tree.data(), query, radius, tree.indices()),
            Self::Clustered => clustered::search(tree, query, radius),
        }
    }
}
