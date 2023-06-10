use rand::prelude::*;
use rayon::prelude::*;

use crate::number::Number;

pub trait Dataset<T: Number, U: Number>: std::fmt::Debug + Send + Sync {
    fn name(&self) -> String;
    fn cardinality(&self) -> usize;
    fn dimensionality(&self) -> usize;
    fn is_metric_expensive(&self) -> bool;
    fn indices(&self) -> &[usize];
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

    // TODO: Clean up the names on these
    fn set_reordered_indices(&mut self, indices: &[usize]);
    // This method will fail if an `reorder` has not been called.
    fn get_reordered_index(&self, i: usize) -> usize;
}
