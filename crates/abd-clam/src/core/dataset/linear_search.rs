//! Traits for datasets that support linear search.

use std::collections::BinaryHeap;

use distances::Number;
use rayon::prelude::*;

use super::{Dataset, ParDataset};

/// A dataset that supports linear search.
pub trait LinearSearch<I, U: Number>: Dataset<I, U> {
    /// Runs linear KNN search on the dataset.
    fn knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let mut knn = SizedHeap::new(Some(k));
        self.query_to_many(query, &indices)
            .into_iter()
            .for_each(|(i, d)| knn.push((d, i)));
        knn.items().map(|(d, i)| (i, d)).collect()
    }

    /// Runs linear RNN search on the dataset.
    fn rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        Dataset::query_to_many(self, query, &indices)
            .into_iter()
            .filter(|&(_, d)| d <= radius)
            .collect()
    }
}

/// An extension of `LinearSearch` that provides parallel implementations of
/// linear search algorithms.
#[allow(clippy::module_name_repetitions)]
pub trait ParLinearSearch<I: Send + Sync, U: Number>: ParDataset<I, U> + LinearSearch<I, U> {
    /// Runs linear KNN search on the dataset, in parallel.
    fn par_knn(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        let mut knn = SizedHeap::new(Some(k));
        self.par_query_to_many(query, &indices)
            .into_iter()
            .for_each(|(i, d)| knn.push((d, i)));
        knn.items().map(|(d, i)| (i, d)).collect()
    }

    /// Runs linear RNN search on the dataset, in parallel.
    fn par_rnn(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        let indices = (0..self.cardinality()).collect::<Vec<_>>();
        self.par_query_to_many(query, &indices)
            .into_par_iter()
            .filter(|&(_, d)| d <= radius)
            .collect()
    }
}

/// A helper struct for maintaining a max heap of a fixed size.
///
/// This is useful for maintaining the `k` nearest neighbors in a search algorithm.
pub struct SizedHeap<T: PartialOrd> {
    /// The heap of items.
    heap: BinaryHeap<MaxItem<T>>,
    /// The maximum size of the heap.
    k: Option<usize>,
}

impl<T: PartialOrd> SizedHeap<T> {
    /// Creates a new `SizedMaxHeap` with a fixed size.
    #[must_use]
    pub fn new(k: Option<usize>) -> Self {
        k.map_or_else(
            || Self {
                heap: BinaryHeap::new(),
                k: None,
            },
            |k| Self {
                heap: BinaryHeap::with_capacity(k),
                k: Some(k),
            },
        )
    }

    /// Returns the maximum size of the heap.
    #[must_use]
    pub const fn k(&self) -> Option<usize> {
        self.k
    }

    /// Pushes an item onto the heap, maintaining the max size.
    pub fn push(&mut self, item: T) {
        if let Some(k) = self.k {
            if self.heap.len() < k {
                self.heap.push(MaxItem(item));
            } else if let Some(top) = self.heap.peek() {
                if item < top.0 {
                    self.heap.pop();
                    self.heap.push(MaxItem(item));
                }
            }
        } else {
            self.heap.push(MaxItem(item));
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

    /// Consumes the `SizedMaxHeap` and returns the items in an iterator.
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
        self.k.map_or(false, |k| self.heap.len() == k)
    }
}

/// A wrapper struct for implementing `PartialOrd` and `Ord` on a type to use
/// with `SizedMaxHeap`.
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
        self.0.partial_cmp(&other.0).unwrap_or(core::cmp::Ordering::Greater)
    }
}
