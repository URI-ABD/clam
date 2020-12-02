use std::{fmt, result};

use dashmap::DashMap;
use ndarray::{Array2, ArrayView1};
use rand::seq::IteratorRandom;
use rayon::prelude::*;

use crate::metric::Metric;
use crate::types::*;

pub struct Dataset {
    pub data: Array2<f64>,
    pub metric: &'static str,
    pub use_cache: bool,
    function: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    cache: DashMap<(Index, Index), f64>,
}

impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Dataset")
            .field("data-shape", &self.data.shape())
            .field("metric", &self.metric)
            .field("cache-usage", &self.use_cache)
            .finish()
    }
}

impl Dataset {
    pub fn new(data: Array2<f64>, metric: &'static str, use_cache: bool) -> Dataset {
        Dataset {
            data,
            metric,
            use_cache,
            function: Metric::on_f64(metric),
            cache: DashMap::new(),
        }
    }

    pub fn indices(&self) -> Indices { (0..self.data.shape()[0]).collect() }

    pub fn nrows(&self) -> usize { self.data.nrows() }

    pub fn clear_cache(&self) { self.cache.clear() }

    pub fn cache_size(&self) -> usize { self.cache.len() }

    #[allow(clippy::ptr_arg)]
    pub fn choose_unique(&self, indices: Indices, n: usize) -> Indices {
        // TODO: actually check for uniqueness among choices
        indices.into_iter().choose_multiple(&mut rand::thread_rng(), n)
    }

    pub fn row(&self, i: Index) -> ArrayView1<f64> { self.data.row(i) }

    pub fn distance(&self, left: Index, right: Index) -> f64 {
        if left == right { 0. }
        else {
            let key = if left < right { (left, right) } else { (right, left) };
            if !self.cache.contains_key(&key) { self.cache.insert(key, (self.function)(self.data.row(left), self.data.row(right))); }
            *self.cache.get(&key).unwrap()
        }
    }

    #[allow(clippy::ptr_arg)]
    pub fn distances_from(&self, left: Index, right: &Indices) -> Vec<f64> {
        right
            .par_iter()
            .map(|&r| self.distance(left, r))
            .collect::<Vec<f64>>()
    }

    #[allow(clippy::ptr_arg)]
    pub fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<f64>> {
        left.par_iter().map(|&l| self.distances_from(l, right)).collect::<Vec<Vec<f64>>>()
    }

    #[allow(clippy::ptr_arg)]
    pub fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<f64>> {
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
        let dataset = Dataset::new(data, "euclidean", false);
        assert_eq!(dataset.nrows(), 2);
        assert_eq!(dataset.row(0), array![1., 2., 3.]);

        approx_eq!(f64, dataset.distance(0, 0), 0.);
        approx_eq!(f64, dataset.distance(0, 1), 3.);
        approx_eq!(f64, dataset.distance(1, 0), 3.);
        approx_eq!(f64, dataset.distance(1, 1), 0.);
    }
}
