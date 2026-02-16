//! Entropy Scaling Nearest Neighbor Search algorithms, and methods for measuring search quality.

use core::borrow::Borrow;

use rayon::prelude::*;

use crate::{DistanceValue, NamedAlgorithm, Tree};

pub mod approximate;
mod exact;
pub mod quality;
pub mod selection;

pub(crate) use exact::knn_bfs;
pub use exact::{KnnBfs, KnnDfs, KnnLinear, KnnRrnn, KnnSieve, RnnChess, RnnLinear};
pub use quality::MeasurableSearchQuality;

/// Entropy Scaling Nearest Neighbor Search algorithms.
///
/// While CAKES stands for "CLAM-Augmented K-nearest neighbors Entropy-scaling Search", the algorithms in this enum include both K-Nearest Neighbors (KNN) and
/// Ranged Nearest Neighbors (RNN) algorithms. The name of the enum is kept as `Cakes` to ensure conceptual continuity from our earlier publications.
///
/// These algorithms are under active development and optimization. While the API is unlikely to change, we will continue to add new algorithms and to optimize
/// the existing ones. We encourage users to experiment with different algorithms to find the best fit for their specific use case.
///
/// In summary, `Cakes` is a collection of search algorithms that leverage the geometric and topological properties of the real-world data to achieve sub-linear
/// scaling of search time with dataset size, and sometimes even scaling independent of dataset size.
///
/// There are two different ways to group these algorithms:
///
/// 1. By the type of search they perform:
///    - K-Nearest Neighbors (KNN) algorithms: These algorithms search for the `k` nearest neighbors of a query.
///    - Ranged Nearest Neighbors (RNN) algorithms: These algorithms search for all neighbors within a certain radius `r` from the query.
/// 2. By whether they are exact or approximate algorithms:
///    - Exact algorithms: These algorithms guarantee perfect recall when the distance function is a metric, i.e. it satisfies the triangle inequality. This
///      means that, under a distance metric, the algorithm will always return the same results as a brute-force linear search. If the distance function is not
///      a metric, recall may be worse.
///    - Approximate algorithms: These algorithms' names are prefixed with "Approx" and they do not guarantee perfect recall. They use a `tol` parameter to
///      control the trade-off between search throughput and various quality measures such as recall and relative distance error. A `tol` of 0 corresponds to an
///      exact algorithm, while higher values of `tol` will end the search earlier, resulting in higher throughput (queries per second) but potentially worse
///      quality.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
#[must_use]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
pub enum Cakes<T: DistanceValue> {
    /// K-Nearest Neighbors Breadth-First Sieve.
    KnnBfs(KnnBfs),
    /// K-Nearest Neighbors Depth-First Sieve.
    KnnDfs(KnnDfs),
    /// K-Nearest Neighbors Linear Search.
    KnnLinear(KnnLinear),
    /// K-Nearest Neighbors Repeated RNN.
    KnnRrnn(KnnRrnn),
    /// K-Nearest Neighbors Sieve.
    KnnSieve(KnnSieve),
    /// Ranged Nearest Neighbors Chess Search.
    RnnChess(RnnChess<T>),
    /// Ranged Nearest Neighbors Linear Search.
    RnnLinear(RnnLinear<T>),
    /// Approximate K-Nearest Neighbors Breadth-First Sieve.
    ApproxKnnBfs(approximate::KnnBfs),
    /// Approximate K-Nearest Neighbors Depth-First Sieve.
    ApproxKnnDfs(approximate::KnnDfs),
    /// Approximate K-Nearest Neighbors Sieve.
    ApproxKnnSieve(approximate::KnnSieve),
}

impl<T: DistanceValue> Cakes<T> {
    /// Returns the corresponding linear version of the algorithm.
    pub const fn linear_variant(self) -> Self {
        match self {
            Self::RnnLinear(_) | Self::KnnLinear(_) => self,
            Self::RnnChess(alg) => Self::RnnLinear(alg.linear_variant()),
            Self::KnnBfs(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::KnnDfs(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::KnnRrnn(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::KnnSieve(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::ApproxKnnBfs(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::ApproxKnnDfs(alg) => Self::KnnLinear(alg.linear_variant()),
            Self::ApproxKnnSieve(alg) => Self::KnnLinear(alg.linear_variant()),
        }
    }
}

/// A Nearest Neighbor Search algorithm.
pub trait Search<Id, I, T, A, M>: NamedAlgorithm
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    /// Searches for nearest neighbors of `query` in the given `tree` and returns a vector of `(index, distance)` pairs into the `items` of the `tree`.
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)>;

    /// Parallel version of [`Self::search`].
    fn par_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync;

    /// Batched version of [`Self::search`].
    fn batch_search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, queries: &[Query]) -> Vec<Vec<(usize, T)>> {
        queries.iter().map(|query| self.search(tree, query)).collect()
    }

    /// Parallel version of [`Self::batch_search`].
    fn par_batch_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(
        &self,
        tree: &Tree<Id, Item, T, A, M>,
        queries: &[Query],
    ) -> Vec<Vec<(usize, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        queries.par_iter().map(|query| self.search(tree, query)).collect()
    }

    /// Parallel batched version of [`Self::batch_search`].
    fn par_batch_par_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(
        &self,
        tree: &Tree<Id, Item, T, A, M>,
        queries: &[Query],
    ) -> Vec<Vec<(usize, T)>>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        queries.par_iter().map(|query| self.par_search(tree, query)).collect()
    }
}

impl<T: DistanceValue> core::fmt::Display for Cakes<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::KnnBfs(alg) => write!(f, "{alg}"),
            Self::KnnDfs(alg) => write!(f, "{alg}"),
            Self::KnnLinear(alg) => write!(f, "{alg}"),
            Self::KnnRrnn(alg) => write!(f, "{alg}"),
            Self::KnnSieve(alg) => write!(f, "{alg}"),
            Self::RnnChess(alg) => write!(f, "{alg}"),
            Self::RnnLinear(alg) => write!(f, "{alg}"),
            Self::ApproxKnnBfs(alg) => write!(f, "{alg}"),
            Self::ApproxKnnDfs(alg) => write!(f, "{alg}"),
            Self::ApproxKnnSieve(alg) => write!(f, "{alg}"),
        }
    }
}

impl<T> NamedAlgorithm for Cakes<T>
where
    T: DistanceValue,
{
    fn name(&self) -> &'static str {
        match self {
            Self::KnnBfs(alg) => alg.name(),
            Self::KnnDfs(alg) => alg.name(),
            Self::KnnLinear(alg) => alg.name(),
            Self::KnnRrnn(alg) => alg.name(),
            Self::KnnSieve(alg) => alg.name(),
            Self::RnnChess(alg) => alg.name(),
            Self::RnnLinear(alg) => alg.name(),
            Self::ApproxKnnBfs(alg) => alg.name(),
            Self::ApproxKnnDfs(alg) => alg.name(),
            Self::ApproxKnnSieve(alg) => alg.name(),
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(
            r"^(knn-bfs|knn-dfs|knn-linear|knn-rrnn|knn-sieve|rnn-chess|rnn-linear|approx-knn-bfs|approx-knn-dfs|approx-knn-sieve)(?:::[a-z]+=\d+(?:\.\d+)?(?:,[a-z]+=\d+(?:\.\d+)?)*)?$"
        )
    }
}

impl<T: DistanceValue> core::str::FromStr for Cakes<T> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::regex_pattern().captures(s).map_or_else(
            || Err(format!("Invalid format for Cakes: {s}")),
            |caps| {
                let algorithm = caps.get(1).map(|m| m.as_str());
                match algorithm {
                    Some("knn-bfs") => KnnBfs::from_str(s).map(Self::from),
                    Some("knn-dfs") => KnnDfs::from_str(s).map(Self::from),
                    Some("knn-linear") => KnnLinear::from_str(s).map(Self::from),
                    Some("knn-rrnn") => KnnRrnn::from_str(s).map(Self::from),
                    Some("knn-sieve") => KnnSieve::from_str(s).map(Self::from),
                    Some("rnn-chess") => RnnChess::from_str(s).map(Self::from),
                    Some("rnn-linear") => RnnLinear::from_str(s).map(Self::from),
                    Some("approx-knn-bfs") => approximate::KnnBfs::from_str(s).map(Self::from),
                    Some("approx-knn-dfs") => approximate::KnnDfs::from_str(s).map(Self::from),
                    Some("approx-knn-sieve") => approximate::KnnSieve::from_str(s).map(Self::from),
                    Some(algorithm) => Err(format!("Unknown Cakes algorithm: {algorithm}. Must be one of knn-bfs, knn-dfs, knn-linear, knn-rrnn, knn-sieve, rnn-chess, rnn-linear, approx-knn-bfs, approx-knn-dfs, or approx-knn-sieve.")),
                    None => Err(format!("Invalid format for Cakes: {s}")),
                }
            },
        )
    }
}

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for Cakes<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn search<Item: Borrow<I>, Query: Borrow<I>>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)> {
        match self {
            Self::KnnBfs(alg) => alg.search(tree, query),
            Self::KnnDfs(alg) => alg.search(tree, query),
            Self::KnnLinear(alg) => alg.search(tree, query),
            Self::KnnRrnn(alg) => alg.search(tree, query),
            Self::KnnSieve(alg) => alg.search(tree, query),
            Self::RnnChess(alg) => alg.search(tree, query),
            Self::RnnLinear(alg) => alg.search(tree, query),
            Self::ApproxKnnBfs(alg) => alg.search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.search(tree, query),
            Self::ApproxKnnSieve(alg) => alg.search(tree, query),
        }
    }

    fn par_search<Item: Borrow<I> + Send + Sync, Query: Borrow<I> + Send + Sync>(&self, tree: &Tree<Id, Item, T, A, M>, query: &Query) -> Vec<(usize, T)>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
    {
        match self {
            Self::KnnBfs(alg) => alg.par_search(tree, query),
            Self::KnnDfs(alg) => alg.par_search(tree, query),
            Self::KnnLinear(alg) => alg.par_search(tree, query),
            Self::KnnRrnn(alg) => alg.par_search(tree, query),
            Self::KnnSieve(alg) => alg.par_search(tree, query),
            Self::RnnChess(alg) => alg.par_search(tree, query),
            Self::RnnLinear(alg) => alg.par_search(tree, query),
            Self::ApproxKnnBfs(alg) => alg.par_search(tree, query),
            Self::ApproxKnnDfs(alg) => alg.par_search(tree, query),
            Self::ApproxKnnSieve(alg) => alg.par_search(tree, query),
        }
    }
}

/// The minimum possible distance from the query to any item in the cluster.
pub(crate) fn d_min<T: DistanceValue>(radius: T, d: T) -> T {
    if d < radius { T::zero() } else { d - radius }
}

/// Returns the theoretical maximum distance from the query to a point in the cluster.
pub(crate) fn d_max<T: DistanceValue>(radius: T, d: T) -> T {
    radius + d
}
