//! Datasets that can be permuted in-place.

use super::Dataset;

/// Datasets that can be permuted in-place.
///
/// This trait extends the `Dataset` trait with the ability to permute the
/// items in the dataset according to a given permutation of indices.
///
/// We provide a blanket implementation of this trait for any type that
/// implements `AsMut<[I]>`, which includes standard collections like `Vec<I>`
/// and slices `[I]`.
pub trait Permutable<I>: Dataset<I> {
    /// Swaps the location of two items in the collection.
    ///
    /// The implementor may choose to panic if either index is out of bounds.
    fn swap(&mut self, a: usize, b: usize);

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
                self.swap(source_index, i);
            }
        }
    }
}

impl<I, D: Dataset<I> + AsMut<[I]>> Permutable<I> for D {
    fn swap(&mut self, a: usize, b: usize) {
        // SAFETY: Since we have &mut self, we have exclusive access to the
        // underlying data. Thus, the pointers ptr_a and ptr_b are guaranteed
        // to be valid and non-overlapping.
        #[allow(unsafe_code)]
        unsafe {
            let ptr_a = self.as_mut().as_mut_ptr().add(a);
            let ptr_b = self.as_mut().as_mut_ptr().add(b);
            std::ptr::swap(ptr_a, ptr_b);
        }
    }
}
