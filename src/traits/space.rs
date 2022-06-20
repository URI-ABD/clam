//! Provides the `Space` trait and a struct `TabularSpace` implementing it.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::dataset::Tabular;
use crate::prelude::*;

/// A `Cache` stores the distance values between pairs of instances as they are
/// computed. This makes it so that no distance value is computed more than
/// once. This can be especially useful when the metric is expensive to compute,
/// as is the case for Levenshtein, Wasserstein, Tanamoto, etc or when
/// individual data instances are too large or expensive to load into memory.
///
/// The implementation of the cache will likely change as we come up with more
/// efficient methods for storing and retrieving distances.
pub type Cache<U> = Arc<RwLock<HashMap<(usize, usize), U>>>;

/// A `Space` represents the combination of a `Dataset` and a `Metric` into a
/// metric space. CLAM is a manifold-mapping framework on such metric spaces.
pub trait Space<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    /// Returns a reference to the underlying dataset.
    fn data(&self) -> &dyn Dataset<T>;

    /// Returns a reference to the underlying metric.
    fn metric(&self) -> &dyn Metric<T, U>;

    /// Whether this `Space` caches distance values to avoid repeatedly
    /// computing the distance between the same instances.
    fn uses_cache(&self) -> bool;

    /// Returns the cache of distance values computed thus far.
    fn cache(&self) -> Cache<U>;

    /// This is built from the names of the dataset and the metric being used.
    fn name(&self) -> String {
        format!("{}__{}", self.data().name(), self.metric().name())
    }

    /// Maps instance indices to a key in the cache.
    fn cache_key(&self, i: usize, j: usize) -> (usize, usize) {
        if i < j {
            (i, j)
        } else {
            (j, i)
        }
    }

    /// Whether the distance between the indexed instances exists in the cache.
    fn is_in_cache(&self, i: usize, j: usize) -> bool {
        let key = self.cache_key(i, j);
        self.cache().read().unwrap().contains_key(&key)
    }

    /// Returns the distance between the two instances from the cache.
    ///
    /// # Panics
    ///
    /// If the distance value is not in the cache. Use `is_in_cache` to avoid.
    fn get_from_cache(&self, i: usize, j: usize) -> U {
        let key = self.cache_key(i, j);
        *self.cache().read().unwrap().get(&key).unwrap()
    }

    /// Store the given distance in the cache. Any previous value will be
    /// overwritten. Returns the newly added value.
    fn add_to_cache(&self, i: usize, j: usize, d: U) -> U {
        let key = self.cache_key(i, j);
        self.cache().write().unwrap().insert(key, d);
        d
    }

    /// Remove a distance value from the cache. Returns the removed value.
    ///
    /// I don't know why you would but here you are.
    ///
    /// # Panics
    ///
    /// If the distance value is not in the cache. Use `is_in_cache` to avoid.
    fn remove_from_cache(&self, i: usize, j: usize) -> U {
        let key = self.cache_key(i, j);
        self.cache().write().unwrap().remove(&key).unwrap()
    }

    /// Empty the cache of all stored values. Returns the number of values that
    /// were removed.
    fn clear_cache(&self) -> usize {
        self.cache().write().unwrap().drain().count()
    }

    /// Two instances are considered equal if the distance between them is zero.
    fn are_instances_equal(&self, left: usize, right: usize) -> bool {
        self.distance_one_to_one(left, right) == U::zero()
    }

    /// Computes/looks-up and returns the distance between two instances.
    fn distance_one_to_one(&self, left: usize, right: usize) -> U {
        // TODO: Refactor to not repeat the distance computation in different if-else blocks.
        if left == right {
            U::zero()
        } else if self.uses_cache() {
            if self.is_in_cache(left, right) {
                self.get_from_cache(left, right)
            } else {
                let left_instance = self.data().get(left);
                let right_instance = self.data().get(right);
                self.add_to_cache(left, right, self.metric().one_to_one(&left_instance, &right_instance))
            }
        } else {
            let left_instance = self.data().get(left);
            let right_instance = self.data().get(right);
            self.metric().one_to_one(&left_instance, &right_instance)
        }
    }

    /// Returns the distances from `left` to each indexed instance in `right`.
    fn distance_one_to_many(&self, left: usize, right: &[usize]) -> Vec<U> {
        if self.metric().is_expensive() || right.len() > 10_000 {
            right.par_iter().map(|&r| self.distance_one_to_one(left, r)).collect()
        } else {
            right.iter().map(|&r| self.distance_one_to_one(left, r)).collect()
        }
    }

    /// Returns the distances from each indexed instance in `left` to each
    /// indexed instance in `right`.
    fn distance_many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<U>> {
        left.iter().map(|&l| self.distance_one_to_many(l, right)).collect()
    }

    /// Returns the all-paris distances between the given indexed instances.
    fn distance_pairwise(&self, indices: &[usize]) -> Vec<Vec<U>> {
        self.distance_many_to_many(indices, indices)
    }

    /// Chooses `n` unique instances from the given indices and returns their
    /// indices.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of unique instances.
    /// * `indices` - Indices from among which to collect sample.
    fn choose_unique(&self, n: usize, indices: &[usize]) -> Vec<usize> {
        let n = if n < indices.len() { n } else { indices.len() };

        let indices = {
            let mut indices = indices.to_vec();
            indices.shuffle(&mut rand::thread_rng());
            indices
        };

        let mut chosen = Vec::new();
        for &i in indices.iter() {
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
}

/// A `Space` for a `Tabular` dataset and an arbitrary `Metric`.
pub struct TabularSpace<'a, T: Number, U: Number> {
    data: &'a Tabular<'a, T>,
    metric: &'a dyn Metric<T, U>,
    uses_cache: bool,
    cache: Cache<U>,
}

impl<'a, T: Number, U: Number> TabularSpace<'a, T, U> {
    /// # Arguments
    ///
    /// * `data` - Reference to a `Tabular` dataset to use in the metric space.
    /// * `metric` - Distance `Metric` to use with the data.
    /// * `use_cache` - Whether to use a `Cache` for avoid repeated distance
    ///                 computations.
    pub fn new(data: &'a Tabular<T>, metric: &'a dyn Metric<T, U>, use_cache: bool) -> TabularSpace<'a, T, U> {
        TabularSpace {
            data,
            metric,
            uses_cache: use_cache,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl<'a, T: Number, U: Number> std::fmt::Debug for TabularSpace<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space")
            .field("data", &self.data.name())
            .field("metric", &self.metric.name())
            .field("uses_cache", &self.uses_cache)
            .finish()
    }
}

impl<'a, T: Number, U: Number> Space<T, U> for TabularSpace<'a, T, U> {
    fn data(&self) -> &dyn Dataset<T> {
        self.data
    }

    fn metric(&self) -> &dyn Metric<T, U> {
        self.metric
    }

    fn uses_cache(&self) -> bool {
        self.uses_cache
    }

    fn cache(&self) -> Cache<U> {
        self.cache.clone()
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use crate::dataset::Tabular;
    use crate::prelude::*;

    use super::TabularSpace;

    #[test]
    fn test_space() {
        let data = vec![vec![1., 2., 3.], vec![3., 3., 1.]];
        let dataset = Tabular::new(&data, "test_space".to_string());
        let metric = metric_from_name("euclidean", false).unwrap();
        let space = TabularSpace::new(&dataset, metric.as_ref(), false);

        approx_eq!(f64, space.distance_one_to_one(0, 0), 0.);
        approx_eq!(f64, space.distance_one_to_one(0, 1), 3.);
        approx_eq!(f64, space.distance_one_to_one(1, 0), 3.);
        approx_eq!(f64, space.distance_one_to_one(1, 1), 0.);
    }
}
