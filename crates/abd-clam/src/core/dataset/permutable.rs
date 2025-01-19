//! A collection whose elements can be permuted in-place.

/// A collection whose elements can be permuted in-place.
///
/// A `Permutable` dataset is useful for our search algorithms, as described in
/// the `CAKES` paper.
///
/// We may *not* want to permute the dataset in-place, e.g. for use with
/// `CHAODA` because it needs to deal with a given set of items under multiple
/// metrics.
pub trait Permutable {
    /// Gets the current permutation of the collection, i.e. the ordering of the
    /// original items into the current order.
    ///
    /// Our implementation of this method on `Vec<T>` and `&mut [T]` will always
    /// return the identity permutation.
    fn permutation(&self) -> Vec<usize>;

    /// Sets the permutation of the collection without modifying the collection.
    fn set_permutation(&mut self, permutation: &[usize]);

    /// Swaps the location of two items in the collection.
    ///
    /// # Arguments
    ///
    /// * `i` - An index in the collection.
    /// * `j` - An index in the collection.
    fn swap_two(&mut self, i: usize, j: usize);

    /// Permutes the collection in-place.
    ///
    /// # Arguments
    ///
    /// * `permutation` - A permutation of the indices of the collection.
    fn permute(&mut self, permutation: &[usize]) {
        // The `source_index` represents the index that we will swap to
        let mut source_index: usize;

        // INVARIANT: After each iteration of the loop, the elements of the
        // sub-array [0..i] are in the correct position.
        for i in 0..(permutation.len() - 1) {
            source_index = permutation[i];

            // Here we're essentially following the cycle. We *know* by
            // the invariant that all elements to the left of i are in
            // the correct position, so what we're doing is following
            // the cycle until we find an index to the right of i. Which,
            // because we followed the position changes, is the correct
            // index to swap.
            while source_index < i {
                source_index = permutation[source_index];
            }

            // If the element at is already at the correct position, we can
            // just skip.
            if source_index != i {
                // We swap to the correct index. Importantly, this index is always
                // to the right of i, we do not modify any index to the left of i.
                // Thus, because we followed the cycle to the correct index to swap,
                // we know that the element at i, after this swap, is in the correct
                // position.
                self.swap_two(source_index, i);
            }
        }
    }
}

impl<T> Permutable for Vec<T> {
    fn permutation(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }

    fn set_permutation(&mut self, _: &[usize]) {}

    fn swap_two(&mut self, i: usize, j: usize) {
        self.swap(i, j);
    }
}

impl<T> Permutable for &mut [T] {
    fn permutation(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }

    fn set_permutation(&mut self, _: &[usize]) {}

    fn swap_two(&mut self, i: usize, j: usize) {
        self.swap(i, j);
    }
}
