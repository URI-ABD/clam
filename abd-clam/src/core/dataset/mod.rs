//! Provides the `Dataset` trait and an implementation for a vector of data.

mod npy_mmap;
mod vec2d;

#[allow(clippy::module_name_repetitions)]
pub use vec2d::VecDataset;

use core::cmp::Ordering;

use rand::prelude::*;
use rayon::prelude::*;

use distances::Number;

/// A common interface for datasets used in CLAM.
pub trait Dataset<T: Send + Sync + Copy, U: Number>: std::fmt::Debug + Send + Sync {
    /// Returns the name of the dataset. This is used to identify the dataset in
    /// various places.
    fn name(&self) -> &str;

    /// Returns the number of instances in the dataset.
    fn cardinality(&self) -> usize;

    /// Whether or not the metric is expensive to calculate.
    ///
    /// If the metric is expensive to calculate, CLAM will enable more parallelism
    /// when calculating distances.
    fn is_metric_expensive(&self) -> bool;

    /// Returns a slice of indices that can be used to access the dataset.
    fn indices(&self) -> &[usize];

    /// Returns the metric used to calculate distances between instances.
    ///
    /// A metric should obey the following properties:
    ///
    /// * Identity: `d(x, y) = 0 <=> x = y`
    /// * Non-negativity: `d(x, y) >= 0`
    /// * Symmetry: `d(x, y) = d(y, x)`
    ///
    /// If the metric also obeys the triangle inequality, `d(x, z) <= d(x, y) + d(y, z)`,
    /// then CLAM can make certain guarantees about the exactness of search results.
    fn metric(&self) -> fn(T, T) -> U;

    /// Swaps the values at two given indices in the dataset.
    ///
    /// Note: It is acceptable for this function to panic if `i` or `j` are not valid indices in the
    /// dataset.
    ///
    /// # Arguments
    ///
    /// * `i` - An index in the dataset.
    /// * `j` - An index in the dataset.
    ///
    /// # Panics
    ///
    /// Implementations of this function may panic if `i` or `j` are not valid indices.
    fn swap(&mut self, i: usize, j: usize);

    /// Sets the reordered indices by a given permutation of indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - A permutation of indices that will be applied to the dataset.
    fn set_reordered_indices(&mut self, indices: &[usize]);

    /// Returns the index of the instance at a given index in the dataset after
    /// reordering.
    ///
    /// # Arguments
    ///
    /// * `i` - An old index in the dataset.
    ///
    /// # Returns
    ///
    /// * The index of the instance at `i` after reordering.
    /// * `None` if the dataset has not been reordered.
    ///
    /// # Panics
    ///
    /// * If `i` is not a valid index in the dataset.
    fn get_reordered_index(&self, i: usize) -> Option<usize>;

    /// Calculates the distance between two indexed instances in the dataset.
    ///
    /// # Arguments
    ///
    /// * `left` - An index in the dataset.
    /// * `right` - An index in the dataset.
    ///
    /// # Returns
    ///
    /// The distance between the instances at `left` and `right`.
    fn one_to_one(&self, left: usize, right: usize) -> U;

    /// Returns whether or not two indexed instances in the dataset are equal.
    ///
    /// As per the definition of a metric, this should return `true` if and only if
    /// the distance between the two instances is zero.
    ///
    /// # Arguments
    ///
    /// * `left` - An index in the dataset
    /// * `right` - An index in the dataset
    ///
    /// # Returns
    ///
    /// `true` if the instances are equal, `false` otherwise
    fn are_instances_equal(&self, left: usize, right: usize) -> bool {
        self.one_to_one(left, right) == U::zero()
    }

    /// Returns a vector of distances.
    ///
    /// # Arguments
    ///
    /// * `left` - An index in the dataset
    /// * `right` - A slice of indices in the dataset
    ///
    /// # Returns
    ///
    /// A vector of distances between the instance at `left` and all instances at `right`
    fn one_to_many(&self, left: usize, right: &[usize]) -> Vec<U> {
        if self.is_metric_expensive() || right.len() > 10_000 {
            right.par_iter().map(|&r| self.one_to_one(left, r)).collect()
        } else {
            right.iter().map(|&r| self.one_to_one(left, r)).collect()
        }
    }

    /// Returns a vector of vectors of distances.
    ///
    /// # Arguments
    ///
    /// * `left` - A slice of indices in the dataset.
    /// * `right` - A slice of indices in the dataset.
    ///
    /// # Returns
    ///
    /// A vector of vectors of distances between the instances at `left` and all instances at `right`
    fn many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<U>> {
        left.iter().map(|&l| self.one_to_many(l, right)).collect()
    }

    /// Returns a vector of distances between all pairs of indexed instances.
    ///
    /// # Arguments
    ///
    /// * `indices` - A slice of indices in the dataset.
    ///
    /// # Returns
    ///
    /// A vector of vectors of distances between all pairs of instances at `indices`
    fn pairwise(&self, indices: &[usize]) -> Vec<Vec<U>> {
        // TODO: Don't repeat the work of having to calculate the metric twice
        // for each pair.
        self.many_to_many(indices, indices)
    }

    /// Calculates the distance between a query and an indexed instance in the dataset.
    ///
    /// # Arguments
    ///
    /// * `query` - A query instance
    /// * `index` - An index in the dataset
    ///
    /// # Returns
    ///
    /// The distance between the query and the instance at `index`
    fn query_to_one(&self, query: T, index: usize) -> U;

    /// Returns a vector of distances between a query and all indexed instances.
    ///
    /// # Arguments
    ///
    /// * `query` - A query instance.
    /// * `indices` - A slice of indices in the dataset.
    ///
    /// # Returns
    ///
    /// A vector of distances between the query and all instances at `indices`
    fn query_to_many(&self, query: T, indices: &[usize]) -> Vec<U> {
        if self.is_metric_expensive() || indices.len() > 1_000 {
            indices
                .par_iter()
                .map(|&index| self.query_to_one(query, index))
                .collect()
        } else {
            indices.iter().map(|&index| self.query_to_one(query, index)).collect()
        }
    }

    /// Chooses a subset of indices that are unique with respect to the metric.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of unique indices to choose.
    /// * `indices` - A slice of indices in the dataset from which to choose.
    /// * `seed` - An optional seed for the random number generator.
    ///
    /// # Returns
    ///
    /// A vector of indices that are unique with respect to the metric. All indices
    /// in the vector are such that no two instances are equal.
    fn choose_unique(&self, n: usize, indices: &[usize], seed: Option<u64>) -> Vec<usize> {
        let n = if n < indices.len() { n } else { indices.len() };

        let indices = {
            let mut indices = indices.to_vec();
            if let Some(seed) = seed {
                indices.shuffle(&mut rand_chacha::ChaCha8Rng::seed_from_u64(seed));
            } else {
                indices.shuffle(&mut rand::thread_rng());
            }
            indices
        };

        let mut chosen = Vec::new();
        for i in indices {
            let is_old = chosen.iter().any(|&o| self.are_instances_equal(i, o));
            if !is_old {
                chosen.push(i);
            }
            if chosen.len() == n {
                break;
            }
        }

        chosen
    }

    /// Reorders the internal dataset by a given permutation of indices.
    ///
    /// # Arguments
    ///
    /// `indices` - A permutation of indices that will be applied to the dataset.
    fn reorder(&mut self, indices: &[usize]) {
        let n = indices.len();

        // TODO: We'll need to support reordering only a subset (i.e. batch)
        // of indices at some point, so this assert will change in the future.

        // The "source index" represents the index that we hope to swap to
        let mut source_index: usize;

        // INVARIANT: After each iteration of the loop, the elements of the
        // subarray [0..i] are in the correct position.
        for i in 0..n - 1 {
            source_index = indices[i];

            // If the element at is already at the correct position, we can
            // just skip.
            if source_index != i {
                // Here we're essentially following the cycle. We *know* by
                // the invariant that all elements to the left of i are in
                // the correct position, so what we're doing is following
                // the cycle until we find an index to the right of i. Which,
                // because we followed the position changes, is the correct
                // index to swap.
                while source_index < i {
                    source_index = indices[source_index];
                }

                // We swap to the correct index. Importantly, this index is always
                // to the right of i, we do not modify any index to the left of i.
                // Thus, because we followed the cycle to the correct index to swap,
                // we know that the element at i, after this swap, is in the correct
                // position.
                self.swap(source_index, i);
            }
        }
        // Inverse mapping
        self.set_reordered_indices(indices);
    }

    /// Calculates the geometric median of a set of indexed instances. Returns
    /// a value from the set of indices that is the index of the median in the
    /// dataset.
    ///
    /// Note: This default implementation does not scale well to arbitrarily large inputs.
    ///
    /// # Arguments
    ///
    /// `indices` - A subset of indices from the dataset
    ///
    /// # Panics
    ///
    /// * If `indices` is empty.
    ///
    /// # Returns
    ///
    /// * The index of the median in the dataset, if `indices` is not empty.
    /// * `None`, if `indices` is empty.
    fn median(&self, indices: &[usize]) -> Option<usize> {
        // TODO: Refactor this to scale for arbitrarily large n
        self.pairwise(indices)
            .into_iter()
            // TODO: Bench using .max instead of .sum
            // .map(|v| v.into_iter().max_by(|l, r| l.partial_cmp(r).unwrap()).unwrap())
            .map(|v| v.into_iter().sum::<U>())
            .enumerate()
            .min_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap_or(Ordering::Greater))
            .map(|(i, _)| indices[i])
    }
}
