use rand::prelude::*;
use rayon::prelude::*;

use crate::core::number::Number;

pub trait Dataset<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    fn name(&self) -> String;
    fn cardinality(&self) -> usize;
    fn dimensionality(&self) -> usize;
    fn is_metric_expensive(&self) -> bool;
    fn indices(&self) -> Vec<usize>;
    fn one_to_one(&self, left: usize, right: usize) -> U;
    fn query_to_one(&self, query: &[T], index: usize) -> U;

    fn are_instances_equal(&self, left: usize, right: usize) -> bool {
        self.one_to_one(left, right) == U::zero()
    }

    fn one_to_many(&self, left: usize, right: &[usize]) -> Vec<U> {
        if self.is_metric_expensive() || right.len() > 10_000 {
            right.par_iter().map(|&r| self.one_to_one(left, r)).collect()
        } else {
            right.iter().map(|&r| self.one_to_one(left, r)).collect()
        }
    }

    fn many_to_many(&self, left: &[usize], right: &[usize]) -> Vec<Vec<U>> {
        left.iter().map(|&l| self.one_to_many(l, right)).collect()
    }

    fn pairwise(&self, indices: &[usize]) -> Vec<Vec<U>> {
        self.many_to_many(indices, indices)
    }

    fn query_to_many(&self, query: &[T], indices: &[usize]) -> Vec<U> {
        if self.is_metric_expensive() || indices.len() > 1_000 {
            indices
                .par_iter()
                .map(|&index| self.query_to_one(query, index))
                .collect()
        } else {
            indices.iter().map(|&index| self.query_to_one(query, index)).collect()
        }
    }

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

pub struct VecVec<T: Number, U: Number> {
    name: String,
    data: Vec<Vec<T>>,
    metric: fn(&[T], &[T]) -> U,
    is_expensive: bool,
}

impl<T: Number, U: Number> VecVec<T, U> {
    pub fn new(data: Vec<Vec<T>>, metric: fn(&[T], &[T]) -> U, name: String, is_expensive: bool) -> Self {
        assert_ne!(data.len(), 0, "Must have some instances in the data.");
        assert_ne!(data[0].len(), 0, "Must have some numbers in the instances.");
        Self {
            name,
            data,
            metric,
            is_expensive,
        }
    }
}

impl<T: Number, U: Number> std::fmt::Debug for VecVec<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        f.debug_struct("Tabular Space").field("name", &self.name).finish()
    }
}

impl<T: Number, U: Number> Dataset<T, U> for VecVec<T, U> {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn cardinality(&self) -> usize {
        self.data.len()
    }

    fn dimensionality(&self) -> usize {
        self.data[0].len()
    }

    fn is_metric_expensive(&self) -> bool {
        self.is_expensive
    }

    fn indices(&self) -> Vec<usize> {
        (0..self.data.len()).collect()
    }

    fn one_to_one(&self, left: usize, right: usize) -> U {
        (self.metric)(&self.data[left], &self.data[right])
    }

    fn query_to_one(&self, query: &[T], index: usize) -> U {
        (self.metric)(query, &self.data[index])
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use crate::utils::distances;

    use super::*;

    #[test]
    fn test_space() {
        let data = vec![vec![1_f32, 2., 3.], vec![3., 3., 1.]];
        let metric = distances::euclidean::<f32, f32>;
        let name = "test".to_string();
        let dataset = VecVec::new(data, metric, name, false);

        approx_eq!(f32, dataset.one_to_one(0, 0), 0.);
        approx_eq!(f32, dataset.one_to_one(0, 1), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 0), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 1), 0.);
    }
}
