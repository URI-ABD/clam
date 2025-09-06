//! A helper struct for maintaining a max heap of an optionally fixed size.

use std::collections::BinaryHeap;

use rayon::prelude::*;

use super::MinItem;

/// A helper struct for maintaining a max heap of a fixed size.
///
/// This is useful for maintaining the `k` nearest neighbors in a search
/// algorithm.
///
/// # Type Parameters
///
/// - `A`: The type of the associated data with each item in the heap. This is
///   ignored when determining the ordering of the heap.
/// - `T`: The type of the items by which the heap is ordered.
#[derive(Debug)]
pub struct SizedHeap<A, T: PartialOrd> {
    /// The heap of items.
    heap: BinaryHeap<MinItem<A, T>>,
    /// The maximum size of the heap.
    k: usize,
}

impl<A, T: PartialOrd> SizedHeap<A, T> {
    /// Creates a new `SizedHeap` with a fixed size.
    #[must_use]
    pub fn new(k: Option<usize>) -> Self {
        k.map_or_else(
            || Self {
                heap: BinaryHeap::new(),
                k: usize::MAX,
            },
            |k| Self {
                heap: BinaryHeap::with_capacity(k),
                k,
            },
        )
    }

    /// Pushes an item onto the heap, maintaining the max size.
    pub fn push(&mut self, (a, item): (A, T)) {
        if self.heap.len() < self.k {
            self.heap.push(MinItem(a, item));
        } else if let Some(top) = self.heap.peek()
            && item < top.1
        {
            self.heap.pop();
            self.heap.push(MinItem(a, item));
        }
    }

    /// Pushes several items onto the heap, maintaining the max size.
    pub fn extend<I: IntoIterator<Item = (A, T)>>(&mut self, items: I) {
        for (a, item) in items {
            self.heap.push(MinItem(a, item));
        }
        while self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    /// Peeks at the top item in the heap.
    #[must_use]
    pub fn peek(&self) -> Option<(&A, &T)> {
        self.heap.peek().map(|MinItem(a, x)| (a, x))
    }

    /// Pops the top item from the heap.
    pub fn pop(&mut self) -> Option<(A, T)> {
        self.heap.pop().map(|MinItem(a, x)| (a, x))
    }

    /// Consumes the `SizedHeap` and returns the items in an iterator.
    pub fn take_items(self) -> impl Iterator<Item = (A, T)> {
        self.heap.into_iter().map(|MinItem(a, x)| (a, x))
    }

    /// Returns whether the heap is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns whether the heap is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    /// Merge two heaps into one.
    pub fn merge(&mut self, other: Self) {
        self.extend(other.take_items());
    }
}

impl<A, T: PartialOrd> FromIterator<(A, T)> for SizedHeap<A, T> {
    fn from_iter<I: IntoIterator<Item = (A, T)>>(iter: I) -> Self {
        let mut heap = Self::new(None);
        for (a, item) in iter {
            heap.push((a, item));
        }
        heap
    }
}

impl<A: Send + Sync, T: PartialOrd + Send + Sync> FromParallelIterator<(A, T)> for SizedHeap<A, T> {
    fn from_par_iter<I: IntoParallelIterator<Item = (A, T)>>(par_iter: I) -> Self {
        par_iter
            .into_par_iter()
            .fold(
                || Self::new(None),
                |mut acc, (a, item)| {
                    acc.push((a, item));
                    acc
                },
            )
            .reduce(
                || Self::new(None),
                |mut acc, heap| {
                    acc.merge(heap);
                    acc
                },
            )
    }
}
