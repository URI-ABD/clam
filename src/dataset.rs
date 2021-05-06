//! `Dataset` trait and some structs implementing it.
//!
//! Contains the declaration and definition of the `Dataset` trait and the
//! `RowMajor` struct implementing Dataset to serves most of the use cases for `CLAM`.
//!
//! TODO: Implement more structs for other types of datasets.
//! For example:
//! * FASTA/FASTQ files containing variable length genomic sequences.
//! * Images. e.g. from SDSS-MaNGA dataset
//! * Molecular graphs with Tanamoto distance.

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Mutex;
use std::{fmt, result};

use ndarray::prelude::*;
use rand::seq::IteratorRandom;
use rayon::prelude::*;

use crate::prelude::*;

/// All datasets supplied to `CLAM` must implement this trait.
///
/// The `cache` for a dataset is optional.
/// It is meant to function as a memo-table and can save significant time
/// when the `Metric` used is computationally expensive.
pub trait Dataset<T, U>: Debug + Send + Sync {
    /// Returns the name of the metric used to compute the distance between instances.
    ///
    /// Warning: This name must be available in the distances crate.
    ///
    /// TODO: change this return type to a closure?
    fn metric(&self) -> Metric<T, U>; // should this return the function directly?

    /// Returns the number of instances in the dataset.
    fn ninstances(&self) -> usize;

    /// Returns the shape of the dataset.
    fn shape(&self) -> &[usize];

    /// Returns the Indices for the dataset.
    fn indices(&self) -> Indices;

    /// Returns the instance at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the instance to return from the dataset.
    ///
    fn instance(&self, index: Index) -> ArrayView<T, IxDyn>;

    /// Returns `n` unique instances from the given indices and returns their indices.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of unique instances
    /// * `indices`:
    ///   - Some - Select unique n from given indices.
    ///   - None - Select unique n from all indices.
    fn choose_unique(&self, indices: Indices, n: usize) -> Indices;

    /// Returns the distance between the two instances at the indices provided.
    ///
    /// # Arguments
    ///
    /// * `left` - Index of an instance to compute distance from
    /// * `right`- Index of an instance to compute distance from
    fn distance(&self, left: Index, right: Index) -> U;

    /// Returns the distances from the instance at left to all instances with indices in right.
    ///
    /// # Arguments
    ///
    /// * `left` - Index of the instance to compute distances from
    /// * `right` - Indices of the instances to compute distances to
    #[allow(clippy::ptr_arg)]
    fn distances_from(&self, left: Index, right: &Indices) -> Vec<U>;

    /// Returns distances from the instances with indices in left to the instances
    /// with indices in right.
    ///
    /// # Arguments
    ///
    /// * `left` - Indices of instances
    /// * `right` - Indices of instances
    #[allow(clippy::ptr_arg)]
    fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<U>>;

    /// Returns the pairwise distances between the instances at the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of instances among which to compute pairwise distances.
    #[allow(clippy::ptr_arg)]
    fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<U>>;

    /// Clears the dataset cache.
    fn clear_cache(&self) {}

    /// Returns the size of the cache used for the dataset.
    ///
    /// If no cache is used, return None.
    /// Otherwise, return the number of entries in the cache.
    fn cache_size(&self) -> Option<usize>;
}

/// RowMajor represents a dataset stored as a 2-dimensional array
/// where rows are instances and columns are features/attributes.
///
/// A wrapper around an `ndarray::Array2` of data, along with a `Metric`,
/// to provide an interface for computing distances between instances
/// contained within the dataset.
///
/// The resulting structure can make use of caching techniques to prevent
/// repeated (potentially expensive) calls to its internal distance function.
pub struct RowMajor<T: Number, U: Number> {
    /// 2D array of data
    pub data: Array2<T>,

    // A str name for the distance function being used
    pub metric_name: &'static str,

    /// Whether this dataset should use an internal cache (recommended)
    pub use_cache: bool,

    // The stored function, used to compute distances.
    pub metric: Metric<T, U>,

    // The internal cache.
    cache: Mutex<HashMap<(Index, Index), U>>,
}

impl<T: Number, U: Number> fmt::Debug for RowMajor<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("RowMajor Dataset")
            .field("data-shape", &self.data.shape())
            .field("metric", &self.metric_name)
            .field("cache-usage", &self.use_cache)
            .finish()
    }
}

impl<T: Number, U: Number> RowMajor<T, U> {
    /// Create a new Dataset, using the provided data and metric, optionally use a cache.
    ///
    /// # Arguments
    ///
    /// * data - a 2-dimensional array.
    /// * name - of distance-metric to use.
    /// * use_cache - whether to use an internal cache for storing distances.
    pub fn new(data: Array2<T>, metric: &'static str, use_cache: bool) -> Result<RowMajor<T, U>, String> {
        Ok(RowMajor {
            data,
            metric_name: metric,
            use_cache,
            metric: metric_new(metric)?,
            cache: Mutex::new(HashMap::new()),
        })
    }
}

impl<T: Number, U: Number> Dataset<T, U> for RowMajor<T, U> {
    /// Return the metric name for the dataset.
    fn metric(&self) -> Metric<T, U> {
        self.metric
    }

    /// Returns the number of rows in the dataset.
    fn ninstances(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the shape of the dataset.
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Return all of the indices in the dataset.
    fn indices(&self) -> Indices {
        (0..self.data.shape()[0]).collect()
    }

    /// Return the row at the provided index.
    fn instance(&self, i: Index) -> ArrayView<T, IxDyn> {
        self.data.index_axis(Axis(0), i).into_dyn()
    }

    /// Return a random selection of unique indices.
    ///
    /// Returns `n` unique random indices from the provided vector.
    /// If `n` is greater than the number of indices provided, the full list is returned in shuffled order.
    #[allow(clippy::ptr_arg)]
    fn choose_unique(&self, indices: Indices, n: usize) -> Indices {
        // TODO: actually check for uniqueness among choices
        indices.into_iter().choose_multiple(&mut rand::thread_rng(), n)
    }

    /// Compute the distance between `left` and `right`.
    fn distance(&self, left: Index, right: Index) -> U {
        if left == right {
            U::zero()
        } else {
            let key = if left < right { (left, right) } else { (right, left) };
            if !self.cache.lock().unwrap().contains_key(&key) {
                let distance = (self.metric)(&self.data.row(left).into_dyn(), &self.data.row(right).into_dyn());
                self.cache.lock().unwrap().insert(key, distance);
                distance
            } else {
                *self.cache.lock().unwrap().get(&key).unwrap()
            }
        }
    }

    /// Compute the distances from `left` to all points in `right`.
    #[allow(clippy::ptr_arg)]
    fn distances_from(&self, left: Index, right: &Indices) -> Vec<U> {
        right.par_iter().map(|&r| self.distance(left, r)).collect::<Vec<U>>()
    }

    /// Compute distances between all points in `left` and `right`.
    #[allow(clippy::ptr_arg)]
    fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<U>> {
        left.par_iter()
            .map(|&l| self.distances_from(l, right))
            .collect::<Vec<Vec<U>>>()
    }

    /// Compute the pairwise distance between all points in `indices`.
    #[allow(clippy::ptr_arg)]
    fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<U>> {
        // TODO: Optimize this to only make distance calls for lower triangular matrix
        self.distances_among(indices, indices)
    }

    /// Clears the internal cache.
    fn clear_cache(&self) {
        self.cache.lock().unwrap().clear()
    }

    /// Returns the number of elements in the internal cache.
    fn cache_size(&self) -> Option<usize> {
        Some(self.cache.lock().unwrap().len())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::prelude::*;

    use super::Dataset;
    use super::RowMajor;

    #[test]
    fn test_dataset() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 3., 1.]];
        let row_0 = array![1., 2., 3.].into_dyn();
        let dataset = RowMajor::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.ninstances(), 2);
        assert_eq!(dataset.instance(0), row_0,);

        approx_eq!(f64, dataset.distance(0, 0), 0.);
        approx_eq!(f64, dataset.distance(0, 1), 3.);
        approx_eq!(f64, dataset.distance(1, 0), 3.);
        approx_eq!(f64, dataset.distance(1, 1), 0.);
    }

    #[test]
    fn test_choose_unique() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 2., 1.]];
        let dataset: RowMajor<f64, f64> = RowMajor::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.choose_unique(vec![0], 1), [0]);
        assert_eq!(dataset.choose_unique(vec![0], 5), [0]);
        assert_eq!(dataset.choose_unique(vec![0, 1], 1).len(), 1);
    }
}
