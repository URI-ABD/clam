use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;

use rand::seq::SliceRandom;
use rayon::prelude::*;

use crate::dataset::Tabular;
use crate::prelude::*;

pub type Cache<U> = Arc<RwLock<HashMap<(usize, usize), U>>>;

pub trait Space<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    fn data(&self) -> &dyn Dataset<T>;

    fn metric(&self) -> &dyn Metric<T, U>;

    fn uses_cache(&self) -> bool;

    fn cache(&self) -> Cache<U>;

    fn name(&self) -> String {
        format!("{}__{}", self.data().name(), self.metric().name())
    }

    fn cache_key(&self, i: usize, j: usize) -> (usize, usize) {
        if i < j {
            (i, j)
        } else {
            (j, i)
        }
    }

    fn is_in_cache(&self, i: usize, j: usize) -> bool {
        let key = self.cache_key(i, j);
        self.cache().read().unwrap().contains_key(&key)
    }

    fn get_from_cache(&self, i: usize, j: usize) -> U {
        let key = self.cache_key(i, j);
        *self.cache().read().unwrap().get(&key).unwrap()
    }

    fn add_to_cache(&self, i: usize, j: usize, d: U) {
        let key = self.cache_key(i, j);
        self.cache().write().unwrap().insert(key, d);
    }

    fn remove_from_cache(&self, i: usize, j: usize) {
        let key = self.cache_key(i, j);
        self.cache().write().unwrap().remove(&key);
    }

    fn clear_cache(&self) -> usize {
        self.cache().write().unwrap().drain().count()
    }

    fn are_instances_equal(&self, left: usize, right: usize) -> bool {
        self.distance_one_to_one(left, right) == U::zero()
    }

    fn distance_one_to_one(&self, left: usize, right: usize) -> U {
        if left == right {
            U::zero()
        } else if self.uses_cache() {
            if self.is_in_cache(left, right) {
                self.get_from_cache(left, right)
            } else {
                let left_instance = self.data().get(left);
                let right_instance = self.data().get(right);
                let d = self.metric().one_to_one(&left_instance, &right_instance);
                self.add_to_cache(left, right, d);
                d
            }
        } else {
            let left_instance = self.data().get(left);
            let right_instance = self.data().get(right);
            self.metric().one_to_one(&left_instance, &right_instance)
        }
    }

    fn distance_one_to_many(&self, left: usize, right: &[usize]) -> Vec<U> {
        if self.metric().is_expensive() || right.len() > 10_000 {
            right.par_iter().map(|&r| self.distance_one_to_one(left, r)).collect()
        } else {
            right.iter().map(|&r| self.distance_one_to_one(left, r)).collect()
        }
    }

    fn distance_many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<U>> {
        left.iter().map(|&l| self.distance_one_to_many(l, right)).collect()
    }

    fn distance_pairwise(&self, indices: &[usize]) -> Vec<Vec<U>> {
        self.distance_many_to_many(indices, indices)
    }

    /// Returns `n` unique instances from the given indices and returns their indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Indices from among which to collect sample.
    /// * `n` - The number of unique instances
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

pub struct TabularSpace<'a, T: Number, U: Number> {
    data: &'a Tabular<'a, T>,
    metric: &'a dyn Metric<T, U>,
    uses_cache: bool,
    cache: Cache<U>,
}

impl<'a, T: Number, U: Number> TabularSpace<'a, T, U> {
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
        f.debug_struct("RowMajor Dataset")
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
        let metric = metric_from_name("euclidean").unwrap();
        let space = TabularSpace::new(&dataset, metric, false);

        approx_eq!(f64, space.distance_one_to_one(0, 0), 0.);
        approx_eq!(f64, space.distance_one_to_one(0, 1), 3.);
        approx_eq!(f64, space.distance_one_to_one(1, 0), 3.);
        approx_eq!(f64, space.distance_one_to_one(1, 1), 0.);
    }
}
