//! The maximum fraction of items that may be in the largest child cluster when partitioning a cluster.

use crate::{DistanceValue, NamedAlgorithm, utils::SizedHeap};

use super::{BipolarSplit, InitialPole, PartitionStrategy};

/// The maximum fraction of items that may be in the largest child cluster when partitioning a cluster.
///
/// For example, under the [`MaxFraction::ThreeQuarters`] strategy for a cluster of 100 items, the largest child cluster can have at most 75 items. If a
/// sub-split contains more than 75 items, it will be split again until all sub-splits have at most 75 items.
///
/// This strategy can be used to prevent highly unbalanced trees, but it may also create much wider trees.
#[must_use]
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub enum MaxFraction {
    /// The maximum fraction of items in the largest child cluster is fixed.
    Fixed(f64),
    /// The maximum fraction of items in the largest child cluster is `9 / 10`.
    #[default]
    NineTenths,
    /// The maximum fraction of items in the largest child cluster is `3 / 4`.
    ThreeQuarters,
}

impl From<MaxFraction> for PartitionStrategy {
    fn from(max_fraction: MaxFraction) -> Self {
        Self::MaxFraction(max_fraction)
    }
}

impl core::fmt::Display for MaxFraction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fixed(k) => write!(f, "max-fraction::fixed={k}"),
            _ => write!(f, "{}", self.name()),
        }
    }
}

impl core::str::FromStr for MaxFraction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let template = Self::regex_pattern();
        if let Some(captures) = template.captures(s) {
            if let Some(fixed) = captures.name("fixed") {
                let fraction = fixed.as_str().parse::<f64>().map_err(|e| format!("Failed to parse fixed fraction: {e}"))?;
                if (0.0..=1.0).contains(&fraction) {
                    Ok(Self::Fixed(fraction))
                } else {
                    Err(format!("Fixed fraction must be between 0 and 1, got {fraction}"))
                }
            } else if s == "max-fraction::nine-tenths" {
                Ok(Self::NineTenths)
            } else if s == "max-fraction::three-quarters" {
                Ok(Self::ThreeQuarters)
            } else {
                Err(format!("Invalid max fraction strategy: {s}. Expected format: {template}"))
            }
        } else {
            Err(format!("Invalid max fraction strategy: {s}. Expected format: {template}"))
        }
    }
}

impl NamedAlgorithm for MaxFraction {
    fn name(&self) -> &'static str {
        match self {
            Self::Fixed(_) => "max-fraction::fixed",
            Self::NineTenths => "max-fraction::nine-tenths",
            Self::ThreeQuarters => "max-fraction::three-quarters",
        }
    }

    fn regex_pattern<'a>() -> &'a lazy_regex::Regex {
        lazy_regex::regex!(r"^max-fraction::(fixed=(?P<fixed>[0-9]*\.?[0-9]+)|nine-tenths|three-quarters)$")
    }
}

impl MaxFraction {
    /// Returns the maximum number of items that can be in the largest child cluster when partitioning a cluster of the given cardinality.
    #[must_use]
    #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn max_items_for(self, cardinality: usize) -> usize {
        let fraction = match self {
            Self::Fixed(fraction) => fraction,
            Self::NineTenths => 0.9,
            Self::ThreeQuarters => 0.75,
        };
        (((cardinality as f64) * fraction).floor() as usize).max(1)
    }

    /// Splits the given items.
    pub(crate) fn split<'a, Id, I, T, M>(
        self,
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

        while splits.peek().is_some_and(|((items, _), (s, _))| items.len() > 1 && *s > max_split_size) {
            // Pop the largest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));

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
        self,
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
        profi::prof!("MaxSplit::par_split");

        // Get the sizes of the left and right splits.
        let nl = l_items.len();
        let nr = r_items.len();
        // Determine the maximum size of the largest split.
        let max_split_size = self.max_items_for(nl + nr + 1);

        // This max-heap will store splits ordered by their size.
        let mut splits = SizedHeap::new(None);
        splits.push(((l_items, l_distances), (nl, 1)));
        splits.push(((r_items, r_distances), (nr, 1 + nl)));

        while splits.peek().is_some_and(|((items, _), (s, _))| items.len() > 1 && *s > max_split_size) {
            // Pop the largest split
            let ((items, distances), (_, ci)) = splits.pop().unwrap_or_else(|| unreachable!("child_items is not empty"));

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
