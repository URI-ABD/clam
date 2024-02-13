//! Algorithms for Ranged Nearest Neighbor search.
//!
//! The stable algorithms are `Linear` and `Clustered`, with the default being `Clustered`.
//!
//! We will experiment with other algorithms in the future, and they will be added to this
//! module as they are being implemented. They should not be considered stable until they
//! are documented as such.

use distances::Number;

use crate::{Cluster, Dataset, Instance, Tree};

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
    pub fn search<I, U, D, C>(self, query: &I, radius: U, tree: &Tree<I, U, D, C>) -> Vec<(usize, U)>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
        C: Cluster<U>,
    {
        match self {
            Self::Linear => {
                let indices = (0..tree.cardinality()).collect::<Vec<_>>();
                linear::search(tree.data(), query, radius, &indices)
            }
            Self::Clustered => clustered::search(tree, query, radius),
        }
    }

    /// Returns the name of the algorithm.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Linear => "Linear",
            Self::Clustered => "Clustered",
        }
    }

    /// Returns the algorithm from a string representation of the name.
    ///
    /// The string is case-insensitive.
    ///
    /// # Arguments
    ///
    /// * `s` - The string representation of the algorithm.
    ///
    /// # Returns
    ///
    /// The algorithm variant.
    ///
    /// # Errors
    ///
    /// If the string does not match any of the algorithms.
    pub fn from_name(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "clustered" => Ok(Self::Clustered),
            _ => Err(format!("Unknown algorithm: {s}")),
        }
    }

    /// Returns a list of all the algorithms, excluding Linear.
    #[must_use]
    pub const fn variants<'a>() -> &'a [Self] {
        &[Self::Clustered]
    }
}
