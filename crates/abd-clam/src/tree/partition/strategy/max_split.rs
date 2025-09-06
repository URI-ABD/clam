//! The maximum fraction of items that must be in the largest child cluster when partitioning a cluster.

use crate::{DistanceValue, utils::SizedHeap};

use super::{BipolarSplit, InitialPole};

/// The maximum fraction of items that must be in the largest child cluster when partitioning a cluster.
#[must_use]
#[derive(Debug, Clone, Copy, Default)]
pub enum MaxSplit {
    /// The maximum fraction of items in the largest child cluster is fixed.
    Fixed(f64),
    /// The maximum fraction of items in the largest child cluster is `9 / 10`.
    NineTenths,
    /// The maximum fraction of items in the largest child cluster is `3 / 4`.
    ThreeQuarters,
    /// No maximum fraction is enforced. This is the default.
    #[default]
    None,
}

impl std::fmt::Display for MaxSplit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(k) => write!(f, "Fixed({k})"),
            Self::NineTenths => write!(f, "NineTenths"),
            Self::ThreeQuarters => write!(f, "ThreeQuarters"),
            Self::None => write!(f, "None"),
        }
    }
}

impl MaxSplit {
    /// Returns the maximum number of items that can be in the largest child cluster when partitioning a cluster of the given cardinality.
    #[must_use]
    #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn max_items_for(&self, cardinality: usize) -> usize {
        let fraction = match self {
            Self::Fixed(fraction) => *fraction,
            Self::NineTenths => 0.9,
            Self::ThreeQuarters => 0.75,
            Self::None => (1_f64).next_down(),
        };
        (((cardinality as f64) * fraction).floor() as usize).max(1)
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
        // Determine the maximum size of the largest split.
        let max_split_size = self.max_items_for(nl + nr + 1);

        // This max-heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(None);
        splits.push(((l_items, l_distances), (nl, 1)));
        splits.push(((r_items, r_distances), (nr, 1 + nl)));

        while splits.peek().is_some_and(|(_, (s, _))| *s > max_split_size) {
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

            // Push the new splits back onto the heap
            splits.push(((l_items, l_distances), (nl, ci)));
            splits.push(((r_items, r_distances), (nr, ci + nl)));
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
        // Determine the maximum size of the largest split.
        let max_split_size = self.max_items_for(nl + nr + 1);

        // This max-heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(None);
        splits.push(((l_items, l_distances), (nl, 1)));
        splits.push(((r_items, r_distances), (nr, 1 + nl)));

        while splits.peek().is_some_and(|(_, (s, _))| *s > max_split_size) {
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
