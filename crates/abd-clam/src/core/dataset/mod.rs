//! Provides the `Dataset` trait and an implementation for a vector of data.

use core::{fmt::Debug, ops::Index};

use std::path::Path;

use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

mod instance;
mod vec2d;

pub use instance::Instance;
#[allow(clippy::module_name_repetitions)]
pub use vec2d::VecDataset;

/// A common interface for datasets used in CLAM.
pub trait Dataset<I: Instance, U: Number>: Debug + Send + Sync + Index<usize, Output = I> {
    /// Changes the metric used to calculate distances between instances.
    ///
    /// This method could potentially be very expensive with memory usage, as it
    /// would likely require cloning the entire dataset.
    #[must_use]
    fn clone_with_new_metric(&self, metric: fn(&I, &I) -> U, is_expensive: bool, name: String) -> Self;

    /// Returns the name of the type of the dataset.
    fn type_name() -> String
    where
        Self: Sized;

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
    fn metric(&self) -> fn(&I, &I) -> U;

    /// Sets the permutation of indices that was used to reorder the dataset.
    ///
    /// This is primarily used when permuting the dataset to reorder it after
    /// building a tree.
    ///
    /// # Arguments
    ///
    /// * `indices` - The permutation of indices.
    fn set_permuted_indices(&mut self, indices: Option<&[usize]>);

    /// Swaps the location of two instances in the dataset.
    ///
    /// This is primarily used when permuting the dataset to reorder it after
    /// building a tree.
    ///
    /// # Arguments
    ///
    /// * `left` - An index in the dataset.
    /// * `right` - An index in the dataset.
    ///
    /// # Errors
    ///
    /// * If there is an error swapping the instances in the implementor.
    ///
    /// # Panics
    ///
    /// * If either `left` or `right` are invalid indices in the dataset.
    fn swap(&mut self, left: usize, right: usize) -> Result<(), String>;

    /// Returns the permutation of indices that was used to reorder the dataset.
    ///
    /// # Returns
    ///
    /// * Some if the dataset was permuted.
    /// * None otherwise.
    fn permuted_indices(&self) -> Option<&[usize]>;

    /// Reorders the internal order of instances by a given permutation of indices.
    ///
    /// # Arguments
    ///
    /// * `permutation` - A permutation of indices in the dataset.
    ///
    /// # Errors
    ///
    /// * See `swap`.
    ///
    /// # Panics
    ///
    /// * If any of the indices in `permutation` are invalid indices in the dataset.
    fn permute_instances(&mut self, permutation: &[usize]) -> Result<(), String> {
        let n = permutation.len();

        // The "source index" represents the index that we hope to swap to
        let mut source_index: usize;

        // INVARIANT: After each iteration of the loop, the elements of the
        // sub-array [0..i] are in the correct position.
        for i in 0..n - 1 {
            source_index = permutation[i];

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
                    source_index = permutation[source_index];
                }

                // We swap to the correct index. Importantly, this index is always
                // to the right of i, we do not modify any index to the left of i.
                // Thus, because we followed the cycle to the correct index to swap,
                // we know that the element at i, after this swap, is in the correct
                // position.
                self.swap(source_index, i)?;
            }
        }

        // Inverse mapping
        self.set_permuted_indices(Some(permutation));

        Ok(())
    }

    /// Get the index before the dataset was reordered. If the dataset was not
    /// reordered, this is the identity function.
    fn original_index(&self, index: usize) -> usize {
        self.permuted_indices().map_or(index, |indices| indices[index])
    }

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
    fn one_to_one(&self, left: usize, right: usize) -> U {
        self.metric()(&self[left], &self[right])
    }

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
        self.query_to_many(&self[left], right)
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

    /// Returns a vector of distances between the given pairs of indexed instances.
    ///
    /// # Arguments
    ///
    /// * `index_pairs` - A slice of pairs of indices in the dataset.
    ///
    /// # Returns
    ///
    /// A vector of distances between the given pairs of instances.
    fn pairs(&self, index_pairs: &[(usize, usize)]) -> Vec<U> {
        if self.is_metric_expensive() {
            index_pairs.par_iter().map(|&(l, r)| self.one_to_one(l, r)).collect()
        } else {
            index_pairs.iter().map(|&(l, r)| self.one_to_one(l, r)).collect()
        }
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
        let n = indices.len();
        let mut matrix = vec![vec![U::zero(); n]; n];

        for (i, &p) in indices.iter().enumerate() {
            let index_pairs = indices.iter().skip(i + 1).map(|&q| (p, q)).collect::<Vec<_>>();
            let distances = self.pairs(&index_pairs);
            distances
                .into_iter()
                .enumerate()
                .map(|(j, d)| (j + i + 1, d))
                .for_each(|(j, d)| {
                    matrix[i][j] = d;
                    matrix[j][i] = d;
                });
        }

        // compute the diagonal for non-metrics
        let index_pairs = indices.iter().map(|&p| (p, p)).collect::<Vec<_>>();
        let distances = self.pairs(&index_pairs);
        distances.into_iter().enumerate().for_each(|(i, d)| matrix[i][i] = d);

        matrix
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
    fn query_to_one(&self, query: &I, index: usize) -> U {
        self.metric()(query, &self[index])
    }

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
    fn query_to_many(&self, query: &I, indices: &[usize]) -> Vec<U> {
        if self.is_metric_expensive() {
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
                indices.shuffle(&mut rand::rngs::StdRng::seed_from_u64(seed));
            } else {
                indices.shuffle(&mut rand::thread_rng());
            }
            indices
        };

        let mut chosen = Vec::new();
        for i in indices {
            // Check if the instance is identical to any of the previously chosen
            // instances.
            let is_chosen = if self.is_metric_expensive() {
                chosen.par_iter().any(|&o| self.are_instances_equal(i, o))
            } else {
                chosen.iter().any(|&o| self.are_instances_equal(i, o))
            };
            if !is_chosen {
                chosen.push(i);
            }
            if chosen.len() == n {
                break;
            }
        }

        chosen
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
        let distances = self
            .pairwise(indices)
            .into_iter()
            // TODO: Bench using .max instead of .sum
            // .map(|v| v.into_iter().max_by(|l, r| l.partial_cmp(r).unwrap()).unwrap())
            .map(|v| v.into_iter().sum::<U>())
            .collect::<Vec<_>>();

        crate::utils::arg_min(&distances).map(|(i, _)| indices[i])
    }

    /// Makes a vector of sharded datasets from the given dataset.
    ///
    /// Each shard will be a random subset of the dataset, and will have a
    /// cardinality of at most `max_cardinality`. The shards will be disjoint
    /// subsets of the dataset.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - The maximum cardinality of each shard.
    fn make_shards(self, max_cardinality: usize) -> Vec<Self>
    where
        Self: Sized;

    /// Saves the dataset to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to save the dataset to.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be saved to the given path.
    fn save(&self, path: &Path) -> Result<(), String>;

    /// Loads a dataset from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the file to load the dataset from.
    /// * `metric` - The metric to use for the dataset.
    /// * `is_expensive` - Whether or not the metric is expensive to calculate.
    ///
    /// # Errors
    ///
    /// * If the dataset cannot be loaded from the given path.
    /// * If the dataset is not the same type as the one that was saved.
    /// * If the file was corrupted.
    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String>
    where
        Self: Sized;

    /// Runs linear KNN search on the dataset.
    fn linear_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let k = k.min(self.cardinality());
        let mut hits = (0..self.cardinality())
            .map(|i| (i, self.query_to_one(query, i)))
            .collect::<Vec<_>>();
        hits.sort_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Greater));
        hits.into_iter().take(k).collect()
    }

    /// Runs parallelized linear KNN search on the dataset.
    fn par_linear_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let k = k.min(self.cardinality());
        let mut hits = (0..self.cardinality())
            .into_par_iter()
            .map(|i| (i, self.query_to_one(query, i)))
            .collect::<Vec<_>>();
        hits.sort_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap_or(std::cmp::Ordering::Greater));
        hits.into_iter().take(k).collect()
    }

    /// Runs linear RNN search on the dataset.
    fn linear_rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        (0..self.cardinality())
            .map(|i| (i, self.query_to_one(query, i)))
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    /// Runs parallelized linear RNN search on the dataset.
    fn par_linear_rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        (0..self.cardinality())
            .into_par_iter()
            .map(|i| (i, self.query_to_one(query, i)))
            .filter(|(_, d)| *d <= radius)
            .collect()
    }
}
