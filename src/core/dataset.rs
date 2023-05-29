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

    /// Swaps the values at two given indices in the dataset.
    ///
    /// Note: It is acceptable for this function to panic if `i` or `j` are not valid indices in the
    /// dataset.
    ///
    /// # Panics
    /// Implementations of this function may panic if `i` or `j` are not valid indices.
    ///
    /// # Arguments
    /// `i` - An index in the dataset
    /// `j` - An index in the dataset
    fn swap(&mut self, i: usize, j: usize);

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

    /// Reorders the internal dataset by a given permutation of indices
    ///
    /// # Arguments
    /// `indices` - A permutation of indices that will be applied to the dataset
    fn reorder(&mut self, indices: &[usize]) {
        let n = indices.len();

        // TODO: We'll need to support reordering only a subset (i.e. batch)
        // of indices at some point, so this assert will change in the future.
        assert!(n == self.cardinality());

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
    }
    /// Calculates the geometric median of a set of indexed instances. Returns
    /// a value from the set of indices that is the index of the median in the
    /// dataset.
    ///
    /// Note: This default implementation does not scale well to arbitrarily large inputs.
    ///
    /// # Panics
    /// This function will panic if given a zero-length slice.
    ///
    /// # Arguments
    /// `indices` - A subset of indices from the dataset
    ///
    /// # Returns
    /// The index of the geometric median of the set of indexed points
    fn median(&self, indices: &[usize]) -> usize {
        // TODO: Refactor this to scale for arbitrarily large n
        indices[self
            .pairwise(indices)
            .into_iter()
            // TODO: Bench using .max instead of .sum
            // .map(|v| v.into_iter().max_by(|l, r| l.partial_cmp(r).unwrap()).unwrap())
            .map(|v| v.into_iter().sum::<U>())
            .enumerate()
            .min_by(|(_, l), (_, r)| l.partial_cmp(r).unwrap())
            .unwrap()
            .0]
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

    fn swap(&mut self, i: usize, j: usize) {
        self.data.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use crate::distances::lp_norms::euclidean;
    use float_cmp::approx_eq;
    use rand::Rng;

    use super::*;

    #[test]
    fn test_space() {
        let data = vec![vec![1_f32, 2., 3.], vec![3., 3., 1.]];
        let metric = euclidean::<f32, f32>;
        let name = "test".to_string();
        let dataset = VecVec::new(data, metric, name, false);

        approx_eq!(f32, dataset.one_to_one(0, 0), 0.);
        approx_eq!(f32, dataset.one_to_one(0, 1), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 0), 3.);
        approx_eq!(f32, dataset.one_to_one(1, 1), 0.);
    }

    #[test]
    fn test_reordering() {
        let mut rng = rand::thread_rng();
        let name = "test".to_string();

        for length in [100, 1_000, 10_000, 100_000] {
            let reference_data: Vec<Vec<f32>> = (0..length)
                .map(|_| {
                    vec![
                        rng.gen_range(0.0..1000.0),
                        rng.gen_range(0.0..1000.0),
                        rng.gen_range(0.0..1000.0),
                    ]
                })
                .collect();

            let mut dataset = VecVec::new(reference_data.clone(), euclidean::<f32, f32>, name.clone(), false);
            let mut new_indices = dataset.indices();
            new_indices.shuffle(&mut rng);

            dataset.reorder(&new_indices);
            for i in 0..length {
                assert_eq!(dataset.data[i], reference_data[new_indices[i]]);
            }
        }
    }
}
