use std::fs::File;
use std::path::PathBuf;
use std::{fmt, result};

use dashmap::DashMap;
use memmap2::Mmap;
use ndarray::prelude::*;
use rand::seq::IteratorRandom;
use rayon::prelude::*;

use crate::prelude::*;

/// RowMajor dataset represented as a 2-dimensional array where rows are instances and columns are attributes.
///
/// A wrapper around an `Array2` of data, along with a provided `metric`, to provide an interface
/// for computing distances between points contained within the dataset.
///
/// The resulting structure can make use of caching techniques to prevent repeated (potentially expensive)
/// calls to its internal distance function.
pub struct RowMajor<T: Number, U: Number> {
    /// 2D array of data
    pub data: Array2<T>,

    // TODO: Remove the string name and rely only on a closure (Metric<T, U>).
    /// Metric to use to compute distances (ex "euclidean")
    pub metric: &'static str,

    /// Whether this dataset should use an internal cache (recommended)
    pub use_cache: bool,

    // The stored function, used to compute distances.
    function: Metric<T, U>,

    // The internal cache.
    cache: DashMap<(Index, Index), U>,
}

impl<T: Number, U: Number> fmt::Debug for RowMajor<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("RowMajor Dataset")
            .field("data-shape", &self.data.shape())
            .field("metric", &self.metric)
            .field("cache-usage", &self.use_cache)
            .finish()
    }
}

impl<T: Number, U: Number> RowMajor<T, U> {
    /// Create a new Dataset, using the provided data and metric, optionally use a cache.
    pub fn new(
        data: Array2<T>,
        metric: &'static str,
        use_cache: bool,
    ) -> Result<RowMajor<T, U>, String> {
        Ok(RowMajor {
            data,
            metric,
            use_cache,
            function: metric_new(metric)?,
            cache: DashMap::new(),
        })
    }
}

impl<T: Number, U: Number> Dataset<T, U> for RowMajor<T, U> {
    /// Return the metric name for the dataset.
    fn metric(&self) -> &'static str {
        self.metric
    }

    /// Returns the number of rows in the dataset.
    fn ninstances(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the shape of the dataset.
    fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
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
        indices
            .into_iter()
            .choose_multiple(&mut rand::thread_rng(), n)
    }

    /// Compute the distance between `left` and `right`.
    fn distance(&self, left: Index, right: Index) -> U {
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
                    (self.function)(
                        &self.data.row(left).into_dyn(),
                        &self.data.row(right).into_dyn(),
                    ),
                );
            }
            *self.cache.get(&key).unwrap()
        }
    }

    /// Compute the distances from `left` to all points in `right`.
    #[allow(clippy::ptr_arg)]
    fn distances_from(&self, left: Index, right: &Indices) -> Vec<U> {
        right
            .par_iter()
            .map(|&r| self.distance(left, r))
            .collect::<Vec<U>>()
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
        self.distances_among(indices, indices)
    }

    /// Clears the internal cache.
    fn clear_cache(&self) {
        self.cache.clear()
    }

    /// Returns the number of elements in the internal cache.
    fn cache_size(&self) -> Option<usize> {
        Some(self.cache.len())
    }
}

pub struct Fasta {
    pub metric: &'static str,
    pub mmap: memmap2::Mmap,
    pub shape: &'static [usize],
    num_sequences: usize,
    seq_len: usize,
    offset: usize,
    function: Metric<u8, u64>,
    cache: DashMap<(Index, Index), u64>,
}

impl fmt::Debug for Fasta {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Fasta Dataset")
            .field("metric", &self.metric)
            .field("offset", &self.offset)
            .field("shape", &self.shape)
            .finish()
    }
}

impl Fasta {
    pub fn new(
        metric: &'static str,
        path: PathBuf,
        shape: &'static [usize],
        offset: usize,
    ) -> Result<Self, String> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return Err("Could not open file".to_string()),
        };
        let mmap = unsafe {
            match Mmap::map(&file) {
                Ok(f) => f,
                Err(_) => return Err("Could not turn file into memmap".to_string()),
            }
        };

        Ok(Fasta {
            metric,
            mmap,
            shape,
            num_sequences: shape[0],
            seq_len: shape[1],
            offset,
            function: metric_new(metric)?,
            cache: DashMap::new(),
        })
    }
}

impl Dataset<u8, u64> for Fasta {
    fn metric(&self) -> &'static str {
        self.metric
    }

    fn ninstances(&self) -> usize {
        self.shape()[0]
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    fn indices(&self) -> Indices {
        (0..self.num_sequences).collect()
    }

    fn instance(&self, index: usize) -> ArrayView<'_, u8, IxDyn> {
        if index >= self.num_sequences {
            panic!("index {} out of range", index)
        };
        let start = self.offset + self.seq_len * index;
        let stop = start + self.seq_len;
        // let instance = Array::from(self.mmap[start..stop].to_vec()).into_dyn().view();
        ArrayView::from_shape([self.seq_len], &self.mmap[start..stop])
            .unwrap()
            .into_dyn()
    }

    fn choose_unique(&self, indices: Indices, n: usize) -> Indices {
        // TODO: actually check for uniqueness among choices
        indices
            .into_iter()
            .choose_multiple(&mut rand::thread_rng(), n)
    }

    // TODO: Figure out some wrapper around indexing for automatic bounds-checking
    fn distance(&self, left: usize, right: usize) -> u64 {
        if left >= self.num_sequences {
            panic!("index {} out of range", left)
        };
        if right >= self.num_sequences {
            panic!("index {} out of range", right)
        };

        if left == right {
            0
        } else {
            let key = if left < right {
                (left, right)
            } else {
                (right, left)
            };
            if !self.cache.contains_key(&key) {
                self.cache.insert(
                    key,
                    (self.function)(&self.instance(left), &self.instance(right)),
                );
            }
            *self.cache.get(&key).unwrap()
        }
    }

    fn distances_from(&self, left: usize, right: &Indices) -> Vec<u64> {
        right
            .par_iter()
            .map(|&r| self.distance(left, r))
            .collect::<Vec<u64>>()
    }

    fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<u64>> {
        left.par_iter()
            .map(|&l| self.distances_from(l, right))
            .collect::<Vec<Vec<u64>>>()
    }

    fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<u64>> {
        self.distances_among(indices, indices)
    }

    fn clear_cache(&self) {
        self.cache.clear()
    }

    fn cache_size(&self) -> Option<usize> {
        Some(self.cache.len())
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;
    use ndarray::prelude::*;

    use crate::prelude::*;
    use crate::sample_datasets::RowMajor;

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

    // #[test]
    // fn test_fasta() {
    //     let mut test_path: PathBuf = [r"/data", "abd", "ann_data"].iter().collect();
    //     test_path.push("silva-SSU-Ref-test");
    //     test_path.set_extension("npy");
    //
    //     let silva_test = Fasta::new(
    //         "hamming",
    //         test_path,
    //         &[10000, 50000],
    //         128,
    //     ).unwrap();
    //
    //     assert_eq!(vec![10000, 50000], silva_test.shape());
    //     assert_eq!(479, silva_test.distance(5, 10));
    //     assert_eq!(0, silva_test.distance(9999, 9999));
    // }

    // #[test]
    // #[should_panic]
    // fn test_fasta_panic() {
    //     let mut test_path: PathBuf = [r"/data", "abd", "ann_data"].iter().collect();
    //     test_path.push("silva-SSU-Ref-test");
    //     test_path.set_extension("npy");
    //
    //     let silva_test = Fasta::new(
    //         "hamming",
    //         test_path,
    //         &[10000, 50000],
    //         128,
    //     ).unwrap();
    //
    //     assert_eq!(vec![10000, 50000], silva_test.shape());
    //     assert_eq!(10000, silva_test.num_sequences);
    //     assert_eq!(50000, silva_test.seq_len);
    //     assert_eq!(0, silva_test.distance(100, 10000));
    // }
}
