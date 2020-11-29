use std::{fmt, result};

use ndarray::{Array1, Array2, ArrayView1, Dim};
use rand::prelude::*;
use rayon::prelude::*;

use crate::metric::Metric;
use crate::types::*;

// TODO: Implement caching of distance values
//  Problem: A global mutable cache feels impossible when there are many distance calls happening in parallel.
pub struct Dataset {
    pub data: Array2<f64>,
    pub metric: &'static str,
    function: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
    // pub cache: HashMap<u64, f64>,
}

impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Dataset")
            .field("data", &self.data)
            .field("metric", &self.metric)
            .finish()
    }
}

impl Dataset {
    pub fn new(data: Array2<f64>, metric: &'static str) -> Dataset {
        Dataset {
            data,
            metric,
            function: Metric::on_f64(metric),
            // cache: HashMap::new(),
        }
    }

    pub fn indices(&self) -> Indices { (0..self.nrows()).collect() }

    pub fn nrows(&self) -> usize { self.data.nrows() }

    #[allow(clippy::ptr_arg)]
    pub fn choose_unique(&self, indices: &Indices, n: usize) -> Indices {
        // TODO: actually check for uniqueness among choices
        let mut rng = &mut rand::thread_rng();
        indices.choose_multiple(&mut rng, n).cloned().collect()
    }

    pub fn row(&self, i: Index) -> ArrayView1<f64> { self.data.row(i) }

    pub fn distance(&self, left: Index, right: Index) -> f64 {
        // TODO: Cache
        (self.function)(self.data.row(left), self.data.row(right))
    }

    #[allow(clippy::ptr_arg)]
    pub fn distances_from(&self, left: Index, right: &Indices) -> Array1<f64> {
        Array1::from(
            right
                .par_iter()
                .map(|&r| self.distance(left, r))
                .collect::<Vec<f64>>()
        )
    }

    #[allow(clippy::ptr_arg)]
    pub fn distances_among(&self, left: &Indices, right: &Indices) -> Array2<f64> {
        let distances = left.par_iter().map(|&l| self.distances_from(l, right)).collect::<Vec<Array1<f64>>>();
        let flattened: Array1<f64> = distances.into_iter().flat_map(|row| row.to_vec()).collect();
        flattened.into_shape(Dim([left.len(), right.len()])).unwrap()
    }

    #[allow(clippy::ptr_arg)]
    pub fn pairwise_distances(&self, indices: &Indices) -> Array2<f64> {
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
        let dataset = Dataset::new(data, "euclidean");
        assert_eq!(dataset.nrows(), 2);
        assert_eq!(dataset.row(0), array![1., 2., 3.]);

        approx_eq!(f64, dataset.distance(0, 0), 0.);
        approx_eq!(f64, dataset.distance(0, 1), 3.);
        approx_eq!(f64, dataset.distance(1, 0), 3.);
        approx_eq!(f64, dataset.distance(1, 1), 0.);
    }
}
