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
use std::sync::Arc;
use std::sync::RwLock;

use ndarray::prelude::*;
use rand::prelude::SliceRandom;
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use sysinfo::System;
use sysinfo::SystemExt;

use crate::metric::metric_new;
use crate::prelude::*;

type Cache<U> = Arc<RwLock<HashMap<(Index, Index), U>>>;

/// All datasets supplied to `CLAM` must implement this trait.
pub trait Dataset<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    /// Returns the function used to compute the distance between instances.
    fn metric(&self) -> Arc<dyn Metric<T, U>>;

    /// Returns the name of the metric in use
    fn metric_name(&self) -> String;

    /// Returns the number of instances in the dataset.
    fn cardinality(&self) -> usize;

    /// Returns the dimensionality of the dataset
    fn dimensionality(&self) -> usize;

    /// Returns the Indices for the dataset.
    fn indices(&self) -> Vec<Index> {
        (0..self.cardinality()).collect()
    }

    /// Returns the instance at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the instance to return from the dataset.
    ///
    fn instance(&self, index: Index) -> Vec<T>;

    /// Returns `n` unique instances from the given indices and returns their indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices from among which to collect sample.
    /// * `n` - The number of unique instances
    fn choose_unique(&self, indices: Vec<Index>, n: usize) -> Vec<Index> {
        // TODO: actually check for uniqueness among choices
        indices.into_iter().choose_multiple(&mut rand::thread_rng(), n)
    }

    /// Randomly sub-samples n unique indices (without replacement) from the dataset.
    /// Returns a tuple of the sub-sampled indices and the remaining indices.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of indices to choose.
    fn subsample_indices(&self, n: usize) -> (Vec<Index>, Vec<Index>) {
        let mut indices = self.indices();
        indices.shuffle(&mut rand::thread_rng());
        let (sample, complement) = indices.split_at(n);
        (sample.to_vec(), complement.to_vec())
    }

    /// Returns the size of the memory footprint of an instance in Bytes.
    fn instance_size(&self) -> usize {
        self.dimensionality() * (U::num_bytes() as usize)
    }

    /// Returns the batch-size to use depending on available RAM.
    ///
    /// # Arguments
    ///
    /// * `memory_fraction` - The fraction (between 0 and 1) of RAM to use for each batch.
    ///                       Defaults to 0.5
    fn batch_size(&self, memory_fraction: Option<f32>) -> usize {
        let batch_fraction = match memory_fraction {
            Some(f) => {
                if 0.0 < f && f < 0.9 {
                    f
                } else {
                    0.5
                }
            }
            None => 0.5,
        };

        let system = System::new_all();
        let available_memory = system.available_memory() as usize;
        let batch_size = (available_memory as f32) * batch_fraction * 1024.;
        (batch_size as usize) / self.instance_size()
    }

    /// Collects the instances corresponding to the given indices and returns them as a RowMajor dataset.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices from which to build the subset
    fn row_major_subset(&self, indices: &[Index]) -> Arc<RowMajor<T, U>> {
        let instances = indices.par_iter().map(|&i| self.instance(i)).collect();
        let subset = RowMajor {
            data: instances,
            metric_name: self.metric_name(),
            use_cache: true,
            metric: self.metric(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        };
        Arc::new(subset)
    }

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
    fn distances_from(&self, left: Index, right: &[Index]) -> Array1<U>;

    /// Returns distances from the instances with indices in left to the instances
    /// with indices in right.
    ///
    /// # Arguments
    ///
    /// * `left` - Indices of instances
    /// * `right` - Indices of instances
    fn distances_among(&self, left: &[Index], right: &[Index]) -> Array2<U>;

    /// Returns the pairwise distances between the instances at the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices of instances among which to compute pairwise distances.
    fn pairwise_distances(&self, indices: &[Index]) -> Array2<U>;
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
    pub data: Vec<Vec<T>>,

    // A str name for the distance function being used
    pub metric_name: String,

    /// Whether this dataset should use an internal cache (recommended)
    pub use_cache: bool,

    // The stored function, used to compute distances.
    pub metric: Arc<dyn Metric<T, U>>,

    // The internal cache.
    cache: Cache<U>,
}

impl<T: Number, U: Number> std::fmt::Debug for RowMajor<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("RowMajor Dataset")
            .field("data-cardinality", &self.cardinality())
            .field("data-dimensionality", &self.dimensionality())
            .field("metric", &self.metric_name)
            .field("cache-usage", &self.use_cache)
            .finish()
    }
}

impl<T: 'static + Number, U: 'static + Number> RowMajor<T, U> {
    /// Create a new Dataset, using the provided data and metric, optionally use a cache.
    ///
    /// # Arguments
    ///
    /// * data - a 2-dimensional array.
    /// * name - of distance-metric to use.
    /// * use_cache - whether to use an internal cache for storing distances.
    pub fn new(data: Vec<Vec<T>>, metric: &str, use_cache: bool) -> Result<RowMajor<T, U>, String> {
        Ok(RowMajor {
            data,
            metric_name: metric.to_string(),
            use_cache,
            metric: metric_new(metric)?,
            cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Clears the internal cache.
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear()
    }

    /// Returns the number of elements in the internal cache.
    pub fn cache_size(&self) -> Option<usize> {
        Some(self.cache.read().unwrap().len())
    }

    pub fn as_arc_dataset(self: Arc<Self>) -> Arc<dyn Dataset<T, U>> {
        self
    }
}

impl<T: Number, U: Number> Dataset<T, U> for RowMajor<T, U> {
    /// Return the Metric used for the dataset.
    fn metric(&self) -> Arc<dyn Metric<T, U>> {
        Arc::clone(&self.metric)
    }

    /// Return the metric name for the dataset.
    fn metric_name(&self) -> String {
        self.metric_name.to_string()
    }

    /// Returns the number of rows in the dataset.
    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn dimensionality(&self) -> usize {
        self.data.par_iter().map(|row| row.len()).max().unwrap()
    }

    /// Return the row at the provided index.
    fn instance(&self, i: Index) -> Vec<T> {
        self.data[i].clone()
    }

    /// Compute the distance between `left` and `right`.
    #[allow(clippy::map_entry)]
    fn distance(&self, left: Index, right: Index) -> U {
        if left == right {
            U::zero()
        } else {
            let key = if left < right { (left, right) } else { (right, left) };
            if !self.cache.read().unwrap().contains_key(&key) {
                let distance = self.metric.distance(&self.instance(left), &self.instance(right));
                self.cache.write().unwrap().insert(key, distance);
                distance
            } else {
                *self.cache.read().unwrap().get(&key).unwrap()
            }
        }
    }

    fn distances_from(&self, left: Index, right: &[Index]) -> Array1<U> {
        Array1::from_vec(right.par_iter().map(|&r| self.distance(left, r)).collect())
    }

    fn distances_among(&self, left: &[Index], right: &[Index]) -> Array2<U> {
        let distances: Array1<U> = left.iter().map(|&l| self.distances_from(l, right).to_vec()).flatten().collect();
        distances.into_shape((left.len(), right.len())).unwrap()
    }

    fn pairwise_distances(&self, indices: &[Index]) -> Array2<U> {
        // TODO: Optimize this to only make distance calls for lower triangular matrix
        self.distances_among(indices, indices)
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use super::Dataset;
    use super::RowMajor;

    #[test]
    fn test_dataset() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let row_0 = vec![1., 2., 3.];
        let dataset = RowMajor::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.cardinality(), 2);
        assert_eq!(dataset.instance(0), row_0);

        approx_eq!(f64, dataset.distance(0, 0), 0.);
        approx_eq!(f64, dataset.distance(0, 1), 3.);
        approx_eq!(f64, dataset.distance(1, 0), 3.);
        approx_eq!(f64, dataset.distance(1, 1), 0.);
    }

    #[test]
    fn test_choose_unique() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let dataset: RowMajor<f64, f64> = RowMajor::new(data, "euclidean", false).unwrap();
        assert_eq!(dataset.choose_unique(vec![0], 1), [0]);
        assert_eq!(dataset.choose_unique(vec![0], 5), [0]);
        assert_eq!(dataset.choose_unique(vec![0, 1], 1).len(), 1);
    }
}
