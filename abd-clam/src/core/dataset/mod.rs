//! Provides the `Dataset` trait and an implementation for a vector of data.

mod instance;
mod vec2d;

pub use instance::Instance;
#[allow(clippy::module_name_repetitions)]
pub use vec2d::VecDataset;

use core::{cmp::Ordering, ops::Index};
use std::path::Path;

use rand::prelude::*;

use distances::Number;

/// A common interface for datasets used in CLAM.
pub trait Dataset<I: Instance, U: Number>: Index<usize, Output = I> + Send + Sync + Sized {
    /// Returns the name of the type of the dataset.
    fn type_name(&self) -> String;

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

    /// Reorders the internal order of instances by a given permutation of indices.
    ///
    /// # Arguments
    ///
    /// * `permutation` - A permutation of indices in the dataset.
    ///
    /// # Errors
    ///
    /// * If any of the indices in `permutation` are invalid indices in the dataset.
    fn permute_instances(&mut self, permutation: &[usize]) -> Result<(), String>;

    /// Returns the permutation of indices that was used to reorder the dataset.
    ///
    /// # Returns
    ///
    /// * Some if the dataset was permuted.
    /// * None otherwise.
    fn permuted_indices(&self) -> Option<&[usize]>;

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
        right.iter().map(|&r| self.one_to_one(left, r)).collect()
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
        indices.iter().map(|&index| self.query_to_one(query, index)).collect()
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

    /// Makes a vector of sharded datasets from the given dataset.
    ///
    /// Each shard will be a random subset of the dataset, and will have a
    /// cardinality of at most `max_cardinality`. The shards will be disjoint
    /// subsets of the dataset.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - The maximum cardinality of each shard.
    fn make_shards(self, max_cardinality: usize) -> Vec<Self>;

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
    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String>;
}
