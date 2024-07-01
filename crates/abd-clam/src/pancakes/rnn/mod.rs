//! Algorithms for Ranged Nearest Neighbor search in a compressed space.

mod clustered;
mod linear;

use distances::number::UInt;

use crate::Instance;

use super::CodecData;

/// The algorithm to use for Ranged Nearest Neighbor search.
pub enum Algorithm {
    /// Use linear search on the entire dataset.
    Linear,
    /// Use a clustered search, as described in the `PanCAKES` paper.
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
    pub fn search<I, U, M>(&self, query: &I, radius: U, data: &CodecData<I, U, M>) -> Vec<(usize, U)>
    where
        I: Instance,
        U: UInt,
        M: Instance,
    {
        match self {
            Self::Linear => linear::search(query, radius, data),
            Self::Clustered => clustered::search(query, radius, data),
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
    pub fn variants() -> Box<[Self]> {
        vec![Self::Clustered].into_boxed_slice()
    }

    /// Returns the baseline algorithm, which is Linear.
    #[must_use]
    pub const fn baseline() -> Self {
        Self::Linear
    }
}
