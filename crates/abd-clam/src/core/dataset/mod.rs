//! Traits relating to datasets.

mod flat_vec;
// pub mod linear_search;
mod metric;
pub mod metric_space;
mod permutable;

use distances::Number;

pub use flat_vec::FlatVec;
// use linear_search::LinearSearch;
pub use metric::Metric;
pub use metric_space::MetricSpace;
pub use permutable::Permutable;

use metric_space::ParMetricSpace;

/// A dataset is a collection of instances.
///
/// # Type Parameters
///
/// - `I`: The type of the instances.
/// - `U`: The type of the distance values.
pub trait Dataset<I, U: Number>: MetricSpace<I, U> {
    /// Returns the name of the dataset.
    fn name(&self) -> &str;

    /// Changes the name of the dataset.
    #[must_use]
    fn with_name(self, name: &str) -> Self;

    /// Returns the number of instances in the dataset.
    fn cardinality(&self) -> usize;

    /// A range of values for the dimensionality of the dataset.
    ///
    /// The first value is the lower bound, and the second value is the upper
    /// bound.
    fn dimensionality_hint(&self) -> (usize, Option<usize>);

    /// Returns the instance at the given index. May panic if the index is out
    /// of bounds.
    fn get(&self, index: usize) -> &I;

    /// Computes the distance between two instances by their indices.
    fn one_to_one(&self, i: usize, j: usize) -> U {
        MetricSpace::one_to_one(self, self.get(i), self.get(j))
    }

    /// Computes the distances between a query instance and the given instance by its index.
    fn query_to_one(&self, query: &I, j: usize) -> U {
        MetricSpace::one_to_one(self, query, self.get(j))
    }

    /// Computes the distances between an instance and a collection of instances.
    fn one_to_many(&self, i: usize, j: &[usize]) -> Vec<(usize, U)> {
        let a = self.get(i);
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        MetricSpace::one_to_many(self, a, &b)
    }

    /// Computes the distances between the query and the given collection of instances.
    fn query_to_many(&self, query: &I, j: &[usize]) -> Vec<(usize, U)> {
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        MetricSpace::one_to_many(self, query, &b)
    }

    /// Computes the distances between two collections of instances.
    fn many_to_many(&self, i: &[usize], j: &[usize]) -> Vec<Vec<(usize, usize, U)>> {
        let a = i.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        MetricSpace::many_to_many(self, &a, &b)
    }

    /// Computes the distances between all given pairs of instances.
    fn pairs(&self, pairs: &[(usize, usize)]) -> Vec<(usize, usize, U)> {
        let pairs = pairs
            .iter()
            .map(|&(i, j)| (i, self.get(i), j, self.get(j)))
            .collect::<Vec<_>>();
        MetricSpace::pairs(self, &pairs)
    }

    /// Computes the distances between all pairs of instances.
    fn pairwise(&self, i: &[usize]) -> Vec<Vec<(usize, usize, U)>> {
        let pairs = i.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        MetricSpace::pairwise(self, &pairs)
    }

    /// Chooses a subset of instances that are unique.
    fn choose_unique(&self, indices: &[usize], choose: usize, seed: Option<u64>) -> Vec<usize> {
        let instances = indices.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        let unique = MetricSpace::choose_unique(self, &instances, choose, seed);
        unique.iter().map(|&(i, _)| i).collect()
    }

    /// Calculates the geometric median of the given instances.
    fn median(&self, indices: &[usize]) -> usize {
        let instances = indices.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        MetricSpace::median(self, &instances).0
    }

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

/// An extension of `Dataset` that provides parallel implementations of
/// distance calculations.
#[allow(clippy::module_name_repetitions)]
pub trait ParDataset<I: Send + Sync, U: Number>: Dataset<I, U> + ParMetricSpace<I, U> {
    /// Parallel version of `one_to_one`.
    fn par_one_to_many(&self, i: usize, j: &[usize]) -> Vec<(usize, U)> {
        let a = self.get(i);
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        ParMetricSpace::par_one_to_many(self, a, &b)
    }

    /// Parallel version of `query_to_one`.
    fn par_query_to_many(&self, query: &I, j: &[usize]) -> Vec<(usize, U)> {
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        ParMetricSpace::par_one_to_many(self, query, &b)
    }

    /// Parallel version of `many_to_many`.
    fn par_many_to_many(&self, i: &[usize], j: &[usize]) -> Vec<Vec<(usize, usize, U)>> {
        let a = i.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        let b = j.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        ParMetricSpace::par_many_to_many(self, &a, &b)
    }

    /// Parallel version of `pairs`.
    fn par_pairs(&self, pairs: &[(usize, usize)]) -> Vec<(usize, usize, U)> {
        let pairs = pairs
            .iter()
            .map(|&(i, j)| (i, self.get(i), j, self.get(j)))
            .collect::<Vec<_>>();
        ParMetricSpace::par_pairs(self, &pairs)
    }

    /// Parallel version of `pairwise`.
    fn par_pairwise(&self, i: &[usize]) -> Vec<Vec<(usize, usize, U)>> {
        let pairs = i.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        ParMetricSpace::par_pairwise(self, &pairs)
    }

    /// Parallel version of `choose_unique`.
    fn par_choose_unique(&self, indices: &[usize], choose: usize, seed: Option<u64>) -> Vec<usize> {
        let instances = indices.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        let unique = ParMetricSpace::par_choose_unique(self, &instances, choose, seed);
        unique.iter().map(|&(i, _)| i).collect()
    }

    /// Parallel version of `median`.
    fn par_median(&self, indices: &[usize]) -> usize {
        let instances = indices.iter().map(|&i| (i, self.get(i))).collect::<Vec<_>>();
        ParMetricSpace::par_median(self, &instances).0
    }

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
            .into_iter()
            .filter(|&(_, d)| d <= radius)
            .collect()
    }
}

/// A helper struct for maintaining a max heap of a fixed size.
///
/// This is useful for maintaining the `k` nearest neighbors in a search algorithm.
pub struct SizedHeap<T: PartialOrd> {
    /// The heap of items.
    heap: std::collections::BinaryHeap<MaxItem<T>>,
    /// The maximum size of the heap.
    k: usize,
}

impl<T: PartialOrd> SizedHeap<T> {
    /// Creates a new `SizedMaxHeap` with a fixed size.
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
        self.heap.len() == self.k
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
