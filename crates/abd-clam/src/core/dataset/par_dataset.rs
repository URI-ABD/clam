//! An extension of the `Dataset` trait for parallel distance computations.

use rayon::prelude::*;

use super::{Dataset, DistanceValue, MaxItem, MinItem};

/// An extension of the `Dataset` trait for parallel distance computations.
///
/// We provide a blanket implementation of this trait for any type that
/// implements `Dataset<I>` and is `Send + Sync`, and where `I` is also
/// `Send + Sync`. This allows for easy use of common collections like `Vec<I>`
/// and slices `&[I]` as parallel datasets.
pub trait ParDataset<I: Send + Sync>: Dataset<I> + Send + Sync {
    /// Returns the distances from a query item to all indexed items in the
    /// given slice.
    fn par_query_to_many<S, T, M>(&self, query: &I, b: S, metric: &M) -> Vec<(usize, T)>
    where
        S: AsRef<[usize]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        b.as_ref()
            .par_iter()
            .map(|&j| (j, metric(query, self.get(j))))
            .collect()
    }

    /// Computes the distances from one indexed item to all indexed items in the
    /// given slice.
    fn par_one_to_many<S, T, M>(&self, a: usize, b: S, metric: &M) -> Vec<(usize, T)>
    where
        S: AsRef<[usize]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        self.par_query_to_many(self.get(a), b, metric)
    }

    /// Computes the pairwise distances between two slices of indexed items.
    fn par_many_to_many<S1, S2, T, M>(&self, a: S1, b: S2, metric: &M) -> Vec<Vec<(usize, usize, T)>>
    where
        S1: AsRef<[usize]>,
        S2: AsRef<[usize]> + Send + Sync,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        a.as_ref()
            .par_iter()
            .map(|&i| {
                b.as_ref()
                    .par_iter()
                    .map(|&j| (i, j, self.one_to_one(i, j, metric)))
                    .collect()
            })
            .collect()
    }

    /// Computes the distances between the given pairs of indexed items.
    fn par_pairs<S, T, M>(&self, pairs: S, metric: &M) -> Vec<(usize, usize, T)>
    where
        S: AsRef<[(usize, usize)]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        pairs
            .as_ref()
            .par_iter()
            .map(|&(i, j)| (i, j, self.one_to_one(i, j, metric)))
            .collect()
    }

    /// Returns the pairwise distance matrix between the given slice of indexed
    /// items.
    fn par_pairwise<S, T, M>(&self, indices: S, metric: &M) -> Vec<Vec<(usize, usize, T)>>
    where
        S: AsRef<[usize]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        let indices = indices.as_ref();
        let n = indices.len();
        let matrix = vec![vec![(0, 0, T::zero()); n]; n];
        indices.par_iter().enumerate().for_each(|(row, &i)| {
            indices.par_iter().enumerate().take(row).for_each(|(col, &j)| {
                let d = self.one_to_one(i, j, metric);
                // SAFETY: We have exclusive access to each row and column
                // because we're iterating in parallel and each (row, col)
                // pair is unique.
                #[allow(unsafe_code)]
                unsafe {
                    let row_ptr = &mut *matrix.as_ptr().cast_mut().add(row);
                    row_ptr[col] = (i, j, d);

                    let col_ptr = &mut *matrix.as_ptr().cast_mut().add(col);
                    col_ptr[row] = (j, i, d);
                }
            });
        });

        let mut matrix = matrix;
        for (i, &index) in (0..n).zip(indices.as_ref()) {
            matrix[i][i] = (index, index, T::zero());
        }

        matrix
    }

    /// Returns the index of the geometric median of the slice of indexed items.
    ///
    /// The geometric median is the item that minimizes the sum of distances to
    /// all other items in the slice.
    fn par_geometric_median<S, T, M>(&self, indices: S, metric: &M) -> usize
    where
        S: AsRef<[usize]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        let distance_matrix = self.par_pairwise(&indices, metric);
        let min_index = distance_matrix
            .into_par_iter()
            .map(|row| row.into_iter().map(|(_, _, d)| d).sum::<T>())
            .enumerate()
            .map(|(i, s)| MinItem(i, s))
            .min_by(Ord::cmp)
            .map_or_else(|| unreachable!("Dataset is empty"), |MinItem(i, _)| i);
        indices.as_ref()[min_index]
    }

    /// Returns the index of the farthest item from the given indexed item
    /// within the slice of indexed items.
    fn par_farthest_among<S, T, M>(&self, a: usize, b: S, metric: &M) -> (usize, T)
    where
        S: AsRef<[usize]>,
        T: DistanceValue + Send + Sync,
        M: (Fn(&I, &I) -> T) + Send + Sync,
    {
        self.par_one_to_many(a, &b, metric)
            .into_iter()
            .map(|(i, d)| MaxItem(i, d))
            .max_by(Ord::cmp)
            .map_or_else(|| unreachable!("Dataset is empty"), |MaxItem(i, d)| (i, d))
    }
}

impl<I: Send + Sync, D: Dataset<I> + Send + Sync> ParDataset<I> for D {}
