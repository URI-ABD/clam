/// CLAM Dataset.
///
/// This module contains the declaration and definition of the Dataset struct.
use std::{fmt, result};

use dashmap::DashMap;
use ndarray::{Array2, ArrayView1};
use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::metric::{metric_new, Metric, Number};
use crate::types::{Index, Indices};

/// Dataset.
///  
/// Datasets wrap an `Array2` of data, along with a `metric`, to provide interfaces
/// for computing distances between points contained within the dataset.
///
/// The resulting structure can make use of caching techniques to prevent repeated (potentially expensive)
/// calls to its internal distance function.
pub struct Dataset<T: Number, U: Number> {
    /// 2D array of data
    pub data: Array2<T>,
    /// Metric to use to compute distances (ex "euclidean")
    pub metric: &'static str,
    /// Whether this dataset should use an internal cache (recommended)
    pub use_cache: bool,

    // The stored function, used to compute distances.
    function: Metric<T, U>,
    // The internal cache.
    cache: DashMap<(Index, Index), U>,
}

impl<T: Number, U: Number> fmt::Debug for Dataset<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Dataset")
            .field("data-shape", &self.data.shape())
            .field("metric", &self.metric)
            .field("cache-usage", &self.use_cache)
            .finish()
    }
}

impl<T: Number, U: Number> Dataset<T, U> {
    /// Create a new Dataset, using the provided data and metric, optionally use a cache.
    pub fn new(
        data: Array2<T>,
        metric: &'static str,
        use_cache: bool,
    ) -> Result<Dataset<T, U>, String> {
        Ok(Dataset {
            data,
            metric,
            use_cache,
            function: metric_new(metric)?,
            cache: DashMap::new(),
        })
    }

    /// Return all of the indices in the dataset.
    pub fn indices(&self) -> Indices {
        (0..self.data.shape()[0]).collect()
    }

    /// Returns the number of rows in the dataset.
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the shape of the dataset.
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Clears the internal cache.
    pub fn clear_cache(&self) {
        self.cache.clear()
    }

    /// Returns the size of the internal cache.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Return a random selection of unique indices.
    /// 
    /// Returns `n` unique random indices from the provided vector.
    /// If `n` is greater than the number of indices provided, the full list is returned in shuffled order.
    #[allow(clippy::ptr_arg)]
    pub fn choose_unique(&self, indices: Indices, n: usize) -> Indices {
        // TODO: actually check for uniqueness among choices
        let mut x = indices;
        x.shuffle(&mut rand::thread_rng());
        x.truncate(n);
        x
    }

    /// Return the row at the provided index.
    pub fn row(&self, i: Index) -> ArrayView1<T> {
        self.data.row(i)
    }

    /// Compute the distance between `left` and `right`.
    pub fn distance(&self, left: Index, right: Index) -> U {
        if left == right {
            U::zero()
        } else {
            let key = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            if !self.cache.contains_key(&key) {
                self.cache.insert(
                    key,
                    (self.function)(self.data.row(left), self.data.row(right)),
                );
            }
            *self.cache.get(&key).unwrap()
        }
    }

    /// Compute the distances from `left` to all points in `right`.
    #[allow(clippy::ptr_arg)]
    pub fn distances_from(&self, left: Index, right: &Indices) -> Vec<U> {
        right
            .par_iter()
            .map(|&r| self.distance(left, r))
            .collect::<Vec<U>>()
    }

    /// Compute distances between all points in `left` and `right`.
    #[allow(clippy::ptr_arg)]
    pub fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<U>> {
        left.par_iter()
            .map(|&l| self.distances_from(l, right))
            .collect::<Vec<Vec<U>>>()
    }

    /// Compute the pairwise distance between all points in `indices`.
    #[allow(clippy::ptr_arg)]
    pub fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<U>> {
        self.distances_among(indices, indices)
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::prelude::*;

    use super::Dataset;

    #[test]
    fn test_dataset() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 3., 1.]];
        let dataset = Dataset::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.nrows(), 2);
        assert_eq!(dataset.row(0), array![1., 2., 3.]);

        approx_eq!(f64, dataset.distance(0, 0), 0.);
        approx_eq!(f64, dataset.distance(0, 1), 3.);
        approx_eq!(f64, dataset.distance(1, 0), 3.);
        approx_eq!(f64, dataset.distance(1, 1), 0.);
    }

    #[test]
    fn test_choose_unique() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 2., 1.]];
        let dataset: Dataset<f64, f64> = Dataset::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.choose_unique(vec![0], 1), [0]);
        assert_eq!(dataset.choose_unique(vec![0], 5), [0]);
        assert_eq!(dataset.choose_unique(vec![0, 1], 1).len(), 1);
    }
}
