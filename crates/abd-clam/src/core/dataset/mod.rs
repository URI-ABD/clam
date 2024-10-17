//! Traits relating to datasets.

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

use super::{metric::ParMetric, Metric};

mod associates_metadata;
mod flat_vec;
mod permutable;
mod sized_heap;

pub use associates_metadata::{AssociatesMetadata, AssociatesMetadataMut};
pub use flat_vec::FlatVec;
pub use permutable::Permutable;
pub use sized_heap::SizedHeap;

#[cfg(feature = "disk-io")]
mod io;

#[cfg(feature = "disk-io")]
#[allow(clippy::module_name_repetitions)]
pub use io::{DatasetIO, ParDatasetIO};

/// A dataset is a collection of items.
///
/// # Type Parameters
///
/// - `I`: The type of the items.
///
/// # Example
///
/// See:
///
/// - [`FlatVec`](crate::core::dataset::FlatVec)
/// - [`CodecData`](crate::pancakes::CodecData)
pub trait Dataset<I> {
    /// Returns the name of the dataset.
    fn name(&self) -> &str;

    /// Changes the name of the dataset.
    #[must_use]
    fn with_name(self, name: &str) -> Self;

    /// Returns the number of items in the dataset.
    fn cardinality(&self) -> usize;

    /// A range of values for the dimensionality of the dataset.
    ///
    /// The first value is the lower bound, and the second value is the upper
    /// bound.
    fn dimensionality_hint(&self) -> (usize, Option<usize>);

    /// Returns the item at the given index. May panic if the index is out
    /// of bounds.
    fn get(&self, index: usize) -> &I;

    /// Returns an iterator over the indices of the items.
    fn indices(&self) -> impl Iterator<Item = usize> {
        0..self.cardinality()
    }

    /// Computes the distance between two items by their indices.
    fn one_to_one<T: Number, M: Metric<I, T>>(&self, i: usize, j: usize, metric: &M) -> T {
        metric.distance(self.get(i), self.get(j))
    }

    /// Computes the distances between an item and a collection of items.
    ///
    /// Each tuple `(j, d)` represents the distance between the items at
    /// indices `i` and `j`.
    fn one_to_many<T: Number, M: Metric<I, T>>(
        &self,
        i: usize,
        js: &[usize],
        metric: &M,
    ) -> impl Iterator<Item = (usize, T)> {
        js.iter().map(move |&j| (j, self.one_to_one(i, j, metric)))
    }

    /// Computes the distance between a query and an item.
    fn query_to_one<T: Number, M: Metric<I, T>>(&self, query: &I, i: usize, metric: &M) -> T {
        metric.distance(query, self.get(i))
    }

    /// Computes the distances between a query and a collection of items.
    ///
    /// Each tuple `(i, d)` represents the distance between the query and the
    /// item at index `i`.
    fn query_to_many<T: Number, M: Metric<I, T>>(
        &self,
        query: &I,
        is: &[usize],
        metric: &M,
    ) -> impl Iterator<Item = (usize, T)> {
        is.iter().map(move |&i| (i, self.query_to_one(query, i, metric)))
    }

    /// Computes the distances between two collections of items.
    ///
    /// Each triplet `(i, j, d)` represents the distance between the items at
    /// indices `i` and `j`.
    fn many_to_many<T: Number, M: Metric<I, T>>(
        &self,
        is: &[usize],
        js: &[usize],
        metric: &M,
    ) -> impl Iterator<Item = Vec<(usize, usize, T)>> {
        is.iter()
            .map(|&i| js.iter().map(|&j| (i, j, self.one_to_one(i, j, metric))).collect())
    }

    /// Computes the distances between all given pairs of items.
    ///
    /// Each triplet `(i, j, d)` represents the distance between the items at
    /// indices `i` and `j`.
    fn pairs<T: Number, M: Metric<I, T>>(
        &self,
        pairs: &[(usize, usize)],
        metric: &M,
    ) -> impl Iterator<Item = (usize, usize, T)> {
        pairs.iter().map(|&(i, j)| (i, j, self.one_to_one(i, j, metric)))
    }

    /// Computes the distances between all pairs of items.
    ///
    /// Each triplet `(i, j, d)` represents the distance between the items at
    /// indices `i` and `j`.
    fn pairwise<T: Number, M: Metric<I, T>>(&self, is: &[usize], metric: &M) -> Vec<Vec<(usize, usize, T)>> {
        if metric.has_symmetry() {
            let mut matrix = is
                .iter()
                .map(|&i| is.iter().map(move |&j| (i, j, T::ZERO)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            for (i, &p) in is.iter().enumerate() {
                let pairs = is.iter().skip(i + 1).map(|&q| (p, q)).collect::<Vec<_>>();
                self.pairs(&pairs, metric)
                    .enumerate()
                    .map(|(j, d)| (j + i + 1, d))
                    .for_each(|(j, (p, q, d))| {
                        matrix[i][j] = (p, q, d);
                        matrix[j][i] = (q, p, d);
                    });
            }

            if !metric.has_identity() {
                // compute the diagonal for non-metrics
                let pairs = is.iter().map(|&p| (p, p)).collect::<Vec<_>>();
                self.pairs(&pairs, metric)
                    .enumerate()
                    .for_each(|(i, (p, q, d))| matrix[i][i] = (p, q, d));
            }

            matrix
        } else {
            self.many_to_many(is, is, metric).collect()
        }
    }

    /// Chooses a subset of items that are unique.
    ///
    /// If the metric has an identity, the first `choose` unique items, i.e.
    /// items that are not equal to any other item, are chosen. Otherwise, a
    /// random subset is chosen.
    fn choose_unique<T: Number, M: Metric<I, T>>(
        &self,
        is: &[usize],
        choose: usize,
        seed: Option<u64>,
        metric: &M,
    ) -> Vec<usize> {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);

        if metric.has_identity() {
            let mut choices = Vec::with_capacity(choose);
            for (i, a) in is.iter().map(|&i| (i, self.get(i))) {
                if !choices.iter().any(|&(_, b)| metric.is_equal(a, b)) {
                    choices.push((i, a));
                }

                if choices.len() == choose {
                    break;
                }
            }
            choices.into_iter().map(|(i, _)| i).collect()
        } else {
            let mut is = is.to_vec();
            is.shuffle(&mut rng);
            is.truncate(choose);
            is
        }
    }

    /// Calculates the geometric median of the given items.
    ///
    /// The geometric median is the item that minimizes the sum of distances
    /// to all other items.
    fn median<T: Number, M: Metric<I, T>>(&self, is: &[usize], metric: &M) -> usize {
        let (arg_median, _) = self
            .pairwise(is, metric)
            .into_iter()
            .map(|row| row.into_iter().map(|(_, _, d)| d).sum::<T>())
            .enumerate()
            .fold(
                (0, T::MAX),
                |(arg_min, min_sum), (i, sum)| {
                    if sum < min_sum {
                        (i, sum)
                    } else {
                        (arg_min, min_sum)
                    }
                },
            );
        is[arg_median]
    }
}

/// Parallel version of [`Dataset`](crate::core::dataset::Dataset).
#[allow(clippy::module_name_repetitions)]
pub trait ParDataset<I: Send + Sync>: Dataset<I> + Send + Sync {
    /// Parallel version of [`Dataset::one_to_one`](crate::core::dataset::Dataset::one_to_one).
    fn par_one_to_one<T: Number, M: ParMetric<I, T>>(&self, i: usize, j: usize, metric: &M) -> T {
        metric.par_distance(self.get(i), self.get(j))
    }

    /// Parallel version of [`Dataset::one_to_many`](crate::core::dataset::Dataset::one_to_many).
    fn par_one_to_many<T: Number, M: ParMetric<I, T>>(
        &self,
        i: usize,
        js: &[usize],
        metric: &M,
    ) -> impl ParallelIterator<Item = (usize, T)> {
        js.par_iter().map(move |&j| (j, self.par_one_to_one(i, j, metric)))
    }

    /// Computes the distance between a query and an item.
    fn par_query_to_one<T: Number, M: ParMetric<I, T>>(&self, query: &I, i: usize, metric: &M) -> T {
        metric.par_distance(query, self.get(i))
    }

    /// Computes the distances between a query and a collection of items.
    ///
    /// Each tuple `(i, d)` represents the distance between the query and the
    /// item at index `i`.
    fn par_query_to_many<T: Number, M: ParMetric<I, T>>(
        &self,
        query: &I,
        is: &[usize],
        metric: &M,
    ) -> impl ParallelIterator<Item = (usize, T)> {
        is.par_iter()
            .map(move |&i| (i, self.par_query_to_one(query, i, metric)))
    }

    /// Parallel version of [`Dataset::many_to_many`](crate::core::dataset::Dataset::many_to_many).
    fn par_many_to_many<T: Number, M: ParMetric<I, T>>(
        &self,
        is: &[usize],
        js: &[usize],
        metric: &M,
    ) -> impl ParallelIterator<Item = Vec<(usize, usize, T)>> {
        is.par_iter().map(|&i| {
            js.par_iter()
                .map(|&j| (i, j, self.par_one_to_one(i, j, metric)))
                .collect()
        })
    }

    /// Parallel version of [`Dataset::pairs`](crate::core::dataset::Dataset::pairs).
    fn par_pairs<T: Number, M: ParMetric<I, T>>(
        &self,
        pairs: &[(usize, usize)],
        metric: &M,
    ) -> impl ParallelIterator<Item = (usize, usize, T)> {
        pairs.par_iter().map(|&(i, j)| (i, j, self.one_to_one(i, j, metric)))
    }

    /// Parallel version of [`Dataset::pairwise`](crate::core::dataset::Dataset::pairwise).
    fn par_pairwise<T: Number, M: ParMetric<I, T>>(&self, a: &[usize], metric: &M) -> Vec<Vec<(usize, usize, T)>> {
        if metric.has_symmetry() {
            let mut matrix = a
                .iter()
                .map(|&i| a.iter().map(move |&j| (i, j, T::ZERO)).collect::<Vec<_>>())
                .collect::<Vec<_>>();

            for (i, &p) in a.iter().enumerate() {
                let pairs = a.iter().skip(i + 1).map(|&q| (p, q)).collect::<Vec<_>>();
                let distances = self.par_pairs(&pairs, metric).collect::<Vec<_>>();
                distances
                    .into_iter()
                    .enumerate()
                    .map(|(j, d)| (j + i + 1, d))
                    .for_each(|(j, (p, q, d))| {
                        matrix[i][j] = (p, q, d);
                        matrix[j][i] = (q, p, d);
                    });
            }

            if !metric.has_identity() {
                // compute the diagonal for non-metrics
                let pairs = a.iter().map(|&p| (p, p)).collect::<Vec<_>>();
                let distances = self.par_pairs(&pairs, metric).collect::<Vec<_>>();
                distances
                    .into_iter()
                    .enumerate()
                    .for_each(|(i, (p, q, d))| matrix[i][i] = (p, q, d));
            }

            matrix
        } else {
            self.par_many_to_many(a, a, metric).collect()
        }
    }

    /// Parallel version of [`Dataset::median`](crate::core::dataset::Dataset::median).
    fn par_median<T: Number, M: ParMetric<I, T>>(&self, is: &[usize], metric: &M) -> usize {
        let (arg_median, _) = self
            .par_pairwise(is, metric)
            .into_iter()
            .map(|row| row.into_iter().map(|(_, _, d)| d).sum::<T>())
            .enumerate()
            .fold(
                (0, T::MAX),
                |(arg_min, min_sum), (i, sum)| {
                    if sum < min_sum {
                        (i, sum)
                    } else {
                        (arg_min, min_sum)
                    }
                },
            );
        is[arg_median]
    }
}
