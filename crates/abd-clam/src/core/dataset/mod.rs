//! Traits for datasets that can be used with CLAM.

use super::DistanceValue;

mod ord_items;
mod par_dataset;
mod permutable;
mod sized_heap;

pub use ord_items::{MaxItem, MinItem};
pub use par_dataset::ParDataset;
pub use permutable::Permutable;
pub use sized_heap::SizedHeap;

/// A trait for datasets that can be used with CLAM.
///
/// A dataset is a collection of items that can be indexed. Each item in the
/// dataset is of type `I`, and the distance between items is computed using a
/// metric function of type `M`. The distance values are of type `T`, which must
/// implement the `DistanceValue` trait.
///
/// In CLAM, we assume that datasets are non-empty and finite. We also assume
/// that `M` is a valid distance measure, meaning it satisfies the following
/// properties for all items `x` and `y` in the dataset:
///
/// 1. Non-negativity: `M(x, y) >= 0`
/// 2. Identity of indiscernibles: `M(x, y) == 0` if and only if `x == y`
/// 3. Symmetry: `M(x, y) == M(y, x)`
///
/// If `M` also satisfies the triangle inequality (`M(x, z) <= M(x, y) + M(y, z)`),
/// then it is a metric in the strict sense, and CLAM can leverage this property
/// to prove exactness of certain algorithms, most notably the search algorithms
/// in the [`cakes`](crate::cakes) and [`pancakes`](crate::pancakes) modules.
///
/// We provide a blanket implementation of this trait for any type that
/// implements `AsRef<[I]>`, which includes standard collections like
/// `Vec<I>` and slices `[I]`.
pub trait Dataset<I> {
    /// Returns a reference to an indexed item from the dataset.
    ///
    /// The implementor may choose to panic if the index is out of bounds.
    fn get(&self, index: usize) -> &I;

    /// Returns the number of items in the dataset.
    fn cardinality(&self) -> usize;

    /// Returns the distance from a query item to the given indexed item.
    fn query_to_one<T: DistanceValue, M: Fn(&I, &I) -> T>(&self, query: &I, b: usize, metric: &M) -> T {
        metric(query, self.get(b))
    }

    /// Returns the distances from a query item to all indexed items in the
    /// given slice.
    fn query_to_many<S: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        query: &I,
        b: S,
        metric: &M,
    ) -> Vec<(usize, T)> {
        b.as_ref().iter().map(|&j| (j, metric(query, self.get(j)))).collect()
    }

    /// Computes the distance between two indexed items in the dataset.
    fn one_to_one<T: DistanceValue, M: Fn(&I, &I) -> T>(&self, a: usize, b: usize, metric: &M) -> T {
        self.query_to_one(self.get(a), b, metric)
    }

    /// Computes the distances from one indexed item to all indexed items in the
    /// given slice.
    fn one_to_many<S: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        a: usize,
        b: S,
        metric: &M,
    ) -> Vec<(usize, T)> {
        self.query_to_many(self.get(a), b, metric)
    }

    /// Computes the pairwise distances between two slices of indexed items.
    fn many_to_many<S1: AsRef<[usize]>, S2: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        a: S1,
        b: S2,
        metric: &M,
    ) -> Vec<Vec<(usize, usize, T)>> {
        a.as_ref()
            .iter()
            .map(|&i| {
                b.as_ref()
                    .iter()
                    .map(|&j| (i, j, self.one_to_one(i, j, metric)))
                    .collect()
            })
            .collect()
    }

    /// Computes the distances between the given pairs of indexed items.
    fn pairs<S: AsRef<[(usize, usize)]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        pairs: S,
        metric: &M,
    ) -> Vec<(usize, usize, T)> {
        pairs
            .as_ref()
            .iter()
            .map(|&(i, j)| (i, j, self.one_to_one(i, j, metric)))
            .collect()
    }

    /// Returns the pairwise distance matrix between the given slice of indexed
    /// items.
    fn pairwise<S: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        indices: S,
        metric: &M,
    ) -> Vec<Vec<(usize, usize, T)>> {
        let indices = indices.as_ref();
        let n = indices.len();
        let mut matrix = vec![vec![(0, 0, T::zero()); n]; n];
        for (row, &i) in indices.iter().enumerate() {
            for (col, &j) in indices.iter().enumerate().take(row) {
                let d = self.one_to_one(i, j, metric);
                matrix[row][col] = (i, j, d);
                matrix[col][row] = (j, i, d);
            }
        }
        matrix
    }

    /// Returns the index of the geometric median of the slice of indexed items.
    ///
    /// The geometric median is the item that minimizes the sum of distances to
    /// all other items in the slice.
    fn geometric_median<S: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        indices: S,
        metric: &M,
    ) -> usize {
        let distance_matrix = self.pairwise(&indices, metric);
        let gm_index = distance_matrix
            .into_iter()
            .map(|row| row.into_iter().map(|(_, _, d)| d).sum::<T>())
            .enumerate()
            .map(|(i, s)| MinItem(i, s))
            .min_by(Ord::cmp)
            .map_or_else(|| unreachable!("Dataset is empty"), |MinItem(i, _)| i);
        indices.as_ref()[gm_index]
    }

    /// Returns the index of the farthest item from the given indexed item
    /// within the slice of indexed items.
    fn farthest_among<S: AsRef<[usize]>, T: DistanceValue, M: Fn(&I, &I) -> T>(
        &self,
        a: usize,
        b: S,
        metric: &M,
    ) -> (usize, T) {
        self.one_to_many(a, &b, metric)
            .into_iter()
            .map(|(i, d)| MaxItem(i, d))
            .max_by(Ord::cmp)
            .map_or_else(|| unreachable!("Dataset is empty"), |MaxItem(i, d)| (i, d))
    }
}

impl<I, D: AsRef<[I]>> Dataset<I> for D {
    fn get(&self, index: usize) -> &I {
        &self.as_ref()[index]
    }

    fn cardinality(&self) -> usize {
        self.as_ref().len()
    }
}
