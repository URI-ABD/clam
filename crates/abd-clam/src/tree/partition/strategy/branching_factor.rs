//! How the number of children is determined when partitioning a cluster.

use crate::{
    DistanceValue,
    utils::{MinItem, SizedHeap},
};

use super::{BipolarSplit, InitialPole};

/// The branching factor determines how many children any given cluster can have when partitioning.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum BranchingFactor {
    /// Each cluster can have zero or `n` children. If `n < 2`, it is treated as 2.
    Fixed(usize),
    /// A cluster with `n` non-center items will have `ceil(log n)` children.
    Logarithmic,
    /// We use some heuristics from our analysis of the expected ratio of the size of the subtree to the cardinality of the cluster, to select a branching
    /// factor that minimizes the expected size of the subtree. This branching factor is recomputed for each cluster based on the number of non-center items in
    /// that cluster.
    Adaptive(usize),
    /// The branching factor is effectively unbounded and will be controlled by the [`SpanReductionFactor`](super::SpanReductionFactor) (SRF).
    #[default]
    Unbounded,
}

impl std::fmt::Display for BranchingFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(k) => write!(f, "Fixed({k})"),
            Self::Logarithmic => write!(f, "Logarithmic"),
            Self::Adaptive(max_k) => write!(f, "Adaptive({max_k})"),
            Self::Unbounded => write!(f, "Unbounded"),
        }
    }
}

impl From<usize> for BranchingFactor {
    fn from(value: usize) -> Self {
        Self::Fixed(value.max(2))
    }
}

impl BranchingFactor {
    /// Returns the branching factor for a cluster with the given the cardinality of the cluster, if not `Unbounded`.
    #[expect(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    #[must_use]
    fn for_cardinality(&self, n: usize) -> usize {
        match self {
            Self::Fixed(b) => *b,
            Self::Logarithmic if n >= 5 => {
                // Use a branching factor of O(log n), where n is the number of non-center items in the cluster.
                // Since `cardinality > 2`, this is at least 2.
                ((n - 1) as f64).log2().ceil() as usize
            }
            Self::Logarithmic => 2, // For n < 5, just use a branching factor of 2
            Self::Adaptive(max_b) => (2..=*max_b)
                .map(|b| (b, expected_num_clusters(n, b) as f64 / n as f64))
                .min_by_key(|&(_, r)| MinItem((), r))
                .map_or(2, |(b, _)| b),
            Self::Unbounded => n - 1, // Effectively no limit on branching factor; SRF will control the actual branching factor
        }
    }

    /// Splits the given items.
    pub(crate) fn split<'a, Id, I, T, M>(
        &self,
        metric: &M,
        l_items: &'a mut [(Id, I)],
        r_items: &'a mut [(Id, I)],
        l_distances: Vec<T>,
        r_distances: Vec<T>,
    ) -> Vec<(usize, &'a mut [(Id, I)])>
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
    {
        // Get the sizes of the left and right splits.
        let nl = l_items.len();
        let nr = r_items.len();
        // Determine the number of splits to create.
        let n_splits = self.for_cardinality(nl + nr + 1);

        // This heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(Some(n_splits));
        splits.push(((l_items, l_distances), (nl, 1)));
        splits.push(((r_items, r_distances), (nr, 1 + nl)));

        while !splits.is_full() {
            // Pop the largest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));
            if items.len() < 2 {
                break;
            }

            // Split it again
            let BipolarSplit {
                l_items,
                r_items,
                l_distances,
                r_distances,
                ..
            } = BipolarSplit::new(items, metric, InitialPole::Distances(distances));

            // Get the sizes and center indices of the new splits
            let nl = l_items.len();
            let nr = r_items.len();
            let lci = ci;
            let rci = ci + nl;

            // Push the new splits back onto the heap
            splits.push(((l_items, l_distances), (nl, lci)));
            splits.push(((r_items, r_distances), (nr, rci)));
        }

        splits.take_items().map(|((c_items, _), (_, ci))| (ci, c_items)).collect::<Vec<_>>()
    }

    /// Parallel version of [`Self::split`].
    pub(crate) fn par_split<'a, Id, I, T, M>(
        &self,
        metric: &M,
        l_items: &'a mut [(Id, I)],
        r_items: &'a mut [(Id, I)],
        l_distances: Vec<T>,
        r_distances: Vec<T>,
    ) -> Vec<(usize, &'a mut [(Id, I)])>
    where
        Id: Send + Sync,
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        // Get the sizes of the left and right splits.
        let nl = l_items.len();
        let nr = r_items.len();
        // Determine the number of splits to create.
        let n_splits = self.for_cardinality(nl + nr + 1);

        // This heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(Some(n_splits));
        splits.push(((l_items, l_distances), (nl, 1)));
        splits.push(((r_items, r_distances), (nr, 1 + nl)));

        while !splits.is_full() {
            // Pop the largest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));
            if items.len() < 2 {
                break;
            }

            // Split it again
            let BipolarSplit {
                l_items,
                r_items,
                l_distances,
                r_distances,
                ..
            } = BipolarSplit::par_new(items, metric, InitialPole::Distances(distances));

            // Get the sizes and center indices of the new splits
            let nl = l_items.len();
            let nr = r_items.len();

            // Push the new splits back onto the heap
            splits.push(((l_items, l_distances), (nl, ci)));
            splits.push(((r_items, r_distances), (nr, ci + nl)));
        }

        splits.take_items().map(|((c_items, _), (_, ci))| (ci, c_items)).collect::<Vec<_>>()
    }
}

/// Recursively finds the number of clusters in a balanced tree with the given branching factor.
///
/// This implements the following recurrence relation:
///   - `T(1) = 1` and `T(2) = 1`, the leaf clusters.
///   - `T(n) = n - 1`, for `3 <= n <= b + 1`, the clusters whose children are all leaves
///   - `T(1 + a + b * n) = 1 + a * T(n + 1) + (b - a) * T(n)`, for `n > b + 1` and `0 <= a < b`.
///
/// This function is used to determine the number of children to create when the [`BranchingFactor`] is `Adaptive`. The chosen branching factor is the value of
/// `b` that minimizes the expected ratio of the size of the subtree to the cardinality of the cluster.
///
/// # Arguments
///
/// - `n`: The cardinality of the cluster (number of items, including the center).
/// - `b`: The branching factor of the tree.
#[must_use]
pub fn expected_num_clusters(n: usize, b: usize) -> usize {
    if n < 3 {
        1
    } else if n < b + 2 {
        n - 1
    } else {
        let a = (n - 1) % b;
        let n = (n - 1) / b;
        1 + a * expected_num_clusters(n + 1, b) + (b - a) * expected_num_clusters(n, b)
    }
}
