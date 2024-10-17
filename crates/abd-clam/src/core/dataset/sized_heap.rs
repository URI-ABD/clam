//! A helper struct for maintaining a max heap of an optionally fixed size.

use rayon::prelude::*;

/// A helper struct for maintaining a max heap of a fixed size.
///
/// This is useful for maintaining the `k` nearest neighbors in a search algorithm.
pub struct SizedHeap<T: PartialOrd> {
    /// The heap of items.
    heap: std::collections::BinaryHeap<MaxItem<T>>,
    /// The maximum size of the heap.
    k: usize,
}

impl<T: PartialOrd> FromIterator<T> for SizedHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut heap = Self::new(None);
        for item in iter {
            heap.push(item);
        }
        heap
    }
}

impl<T: PartialOrd> SizedHeap<T> {
    /// Creates a new `SizedHeap` with a fixed size.
    #[must_use]
    pub fn new(k: Option<usize>) -> Self {
        k.map_or_else(
            || Self {
                heap: std::collections::BinaryHeap::new(),
                k: usize::MAX,
            },
            |k| Self {
                heap: std::collections::BinaryHeap::with_capacity(k),
                k,
            },
        )
    }

    /// Returns the maximum size of the heap.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Pushes an item onto the heap, maintaining the max size.
    pub fn push(&mut self, item: T) {
        if self.heap.len() < self.k {
            self.heap.push(MaxItem(item));
        } else if let Some(top) = self.heap.peek() {
            if item < top.0 {
                self.heap.pop();
                self.heap.push(MaxItem(item));
            }
        }
    }

    /// Pushes several items onto the heap, maintaining the max size.
    pub fn extend<I: Iterator<Item = T>>(&mut self, items: I) {
        for item in items {
            self.heap.push(MaxItem(item));
        }
        while self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    /// Peeks at the top item in the heap.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|MaxItem(x)| x)
    }

    /// Pops the top item from the heap.
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|MaxItem(x)| x)
    }

    /// Consumes the `SizedHeap` and returns the items in an iterator.
    pub fn items(self) -> impl Iterator<Item = T> {
        self.heap.into_iter().map(|MaxItem(x)| x)
    }

    /// Returns the number of items in the heap.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns whether the heap is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns whether the heap is full.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.heap.len() == self.k
    }

    /// Merge two heaps into one.
    pub fn merge(&mut self, other: Self) {
        self.extend(other.items());
    }

    /// Retains only the elements that satisfy the predicate.
    pub fn retain<F: Fn(&T) -> bool>(&mut self, f: F) {
        self.heap.retain(|MaxItem(x)| f(x));
    }
}

impl<T: PartialOrd + Send + Sync> SizedHeap<T> {
    /// Pushes several items onto the heap, maintaining the max size.
    pub fn par_extend<I: ParallelIterator<Item = T>>(&mut self, items: I) {
        for item in items.collect::<Vec<_>>() {
            self.heap.push(MaxItem(item));
        }
        while self.heap.len() > self.k {
            self.heap.pop();
        }
    }

    /// Parallel version of [`SizedHeap::items`](crate::core::dataset::SizedHeap::items).
    #[must_use]
    pub fn par_items(self) -> impl ParallelIterator<Item = T> {
        self.heap.into_par_iter().map(|MaxItem(x)| x)
    }
}

/// A wrapper struct for implementing `PartialOrd` and `Ord` on a type to use
/// with `SizedHeap`.
struct MaxItem<T: PartialOrd>(T);

impl<T: PartialOrd> PartialEq for MaxItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: PartialOrd> Eq for MaxItem<T> {}

impl<T: PartialOrd> PartialOrd for MaxItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialOrd> Ord for MaxItem<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(core::cmp::Ordering::Less)
    }
}
