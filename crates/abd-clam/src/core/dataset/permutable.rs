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
    /// Gets a mutable reference to the item at the specified index.
    fn get_mut(&mut self, index: usize) -> &mut I;

    /// Permutes the collection in-place.
    ///
    /// # Arguments
    ///
    /// * `permutation` - A permutation of the indices of the collection.
    fn permute<S: AsRef<[usize]>>(&mut self, permutation: S) {
        let permutation = permutation.as_ref();

        // The `source_index` represents the index that we will swap to
        let mut source_index: usize;

        // INVARIANT: After each iteration of the loop, the elements of the
        // sub-array `0..i` are in the correct position.
        for i in 0..(permutation.len() - 1) {
            source_index = permutation[i];

            // Here we're essentially following the cycle. We *know* by
            // the invariant that all elements to the left of `i` are in
            // the correct position, so what we're doing is following
            // the cycle until we find an index to the right of `i`. Which,
            // because we followed the position changes, is the correct
            // index to swap.
            while source_index < i {
                source_index = permutation[source_index];
            }

            // If the element at is already at the correct position, we can
            // just skip.
            if source_index != i {
                // We swap to the correct index. Importantly, this index is
                // always to the right of `i`, we do not modify any index to the
                // left of `i`. Thus, because we followed the cycle to the
                // correct index to swap, we know that the element at `i`, after
                // this swap, is in the correct position.

                let ptr_a = self.get_mut(i) as *mut I;
                let ptr_b = self.get_mut(source_index) as *mut I;
                // SAFETY: Since we have &mut self, we have exclusive access to
                // the underlying data. We have also checked that `source_index`
                // and `i` are different, so the pointers `ptr_a` and `ptr_b`
                // are guaranteed to be valid and non-overlapping.
                #[allow(unsafe_code)]
                unsafe {
                    std::ptr::swap(ptr_a, ptr_b);
                }
            }
        }
    }
}

impl<I, D: Dataset<I> + AsMut<[I]>> Permutable<I> for D {
    fn get_mut(&mut self, index: usize) -> &mut I {
        &mut self.as_mut()[index]
    }
}
