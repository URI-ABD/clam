//! Algorithms for Ranged Nearest Neighbor search.
//!
//! The stable algorithms are `Linear` and `Clustered`, with the default being `Clustered`.
//!
//! We will experiment with other algorithms in the future, and they will be added to this
//! module as they are being implemented. They should not be considered stable until they
//! are documented as such.

use distances::Number;

use crate::{Cluster, Dataset, Tree};

/// The algorithm to use for Ranged Nearest Neighbor search.
///
/// The default is `Clustered`, as determined by the benchmarks in the crate.
#[allow(clippy::module_name_repetitions)]
#[derive(Clone, Copy, Debug)]
pub enum RnnAlgorithm {
    /// Use linear search on the entire dataset. This is a stable algorithm.
    Linear,
    /// Use a clustered search, as described in the CHESS paper. This is a
    /// stable algorithm.
    Clustered,
}

impl Default for RnnAlgorithm {
    fn default() -> Self {
        Self::Clustered
    }
}

impl RnnAlgorithm {
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
            Self::Linear => Self::linear_search(tree.data(), query, radius, tree.indices()),
            Self::Clustered => Self::clustered_search(tree, query, radius),
        }
    }

    /// Linear search for the nearest neighbors of a query.
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
    pub(crate) fn linear_search<T, U, D>(data: &D, query: T, radius: U, indices: &[usize]) -> Vec<(usize, U)>
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

    /// Clustered search for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `tree` - The tree to search.
    /// * `query` - The query to search around.
    /// * `radius` - The radius to search within.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn clustered_search<T, U, D>(tree: &Tree<T, U, D>, query: T, radius: U) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let [confirmed, straddlers] = Self::tree_search(tree.data(), tree.root(), query, radius);
        Self::leaf_search(tree.data(), confirmed, straddlers, query, radius)
    }

    /// Perform coarse-grained tree search.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `root` - The root of the tree to search.
    /// * `query` - The query to search around.
    /// * `radius` - The radius to search within.
    ///
    /// # Returns
    ///
    /// A 2-slice of vectors of 2-tuples, where the first element in the slice
    /// is the confirmed clusters, i.e. those that are contained within the
    /// query ball, and the second element is the straddlers, i.e. those that
    /// overlap the query ball. The 2-tuples are the clusters and the distance
    /// from the query to the cluster center.
    pub(crate) fn tree_search<'a, T, U, D>(
        data: &D,
        root: &'a Cluster<T, U>,
        query: T,
        radius: U,
    ) -> [Vec<(&'a Cluster<T, U>, U)>; 2]
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let mut confirmed = Vec::new();
        let mut straddlers = Vec::new();
        let mut candidates = vec![root];

        let (mut terminal, mut non_terminal): (Vec<_>, Vec<_>);
        while !candidates.is_empty() {
            (terminal, non_terminal) = candidates
                .into_iter()
                .map(|c| (c, c.distance_to_instance(data, query)))
                .filter(|&(c, d)| d <= (c.radius + radius))
                .partition(|&(c, d)| (c.radius + d) <= radius);
            confirmed.append(&mut terminal);

            (terminal, non_terminal) = non_terminal.into_iter().partition(|&(c, _)| c.is_leaf());
            straddlers.append(&mut terminal);

            candidates = non_terminal
                .into_iter()
                .flat_map(|(c, d)| {
                    if d < c.radius {
                        c.overlapping_children(data, query, radius)
                    } else {
                        c.children().unwrap().to_vec()
                    }
                })
                .collect();
        }

        [confirmed, straddlers]
    }

    /// Perform fine-grained leaf search
    pub(crate) fn leaf_search<T, U, D>(
        data: &D,
        confirmed: Vec<(&Cluster<T, U>, U)>,
        straddlers: Vec<(&Cluster<T, U>, U)>,
        query: T,
        radius: U,
    ) -> Vec<(usize, U)>
    where
        T: Send + Sync + Copy,
        U: Number,
        D: Dataset<T, U>,
    {
        let hits = confirmed.into_iter().flat_map(|(c, d)| {
            let distances = if c.is_singleton() {
                vec![d; c.cardinality]
            } else {
                data.query_to_many(query, c.indices(data))
            };
            c.indices(data).iter().copied().zip(distances.into_iter())
        });

        let indices = straddlers
            .into_iter()
            .flat_map(|(c, _)| c.indices(data))
            .copied()
            .collect::<Vec<_>>();

        hits.chain(Self::linear_search(data, query, radius, &indices).into_iter())
            .collect()
    }
}
