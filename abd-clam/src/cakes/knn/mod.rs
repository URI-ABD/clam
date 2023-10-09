//! Algorithms for K Nearest Neighbor search.
//!
//! The stable algorithms are `Linear` and `RepeatedRnn`, with the default being
//! `RepeatedRnn`.
//!
//! We will experiment with other algorithms in the future, and they will be added
//! to this enum as they are being implemented. They should not be considered
//! stable until they are documented as such.

use core::{cmp::Ordering, hash::Hash};

use distances::Number;
use priority_queue::PriorityQueue;

use crate::{Dataset, Instance, Tree};

pub(crate) mod greedy_sieve;
pub(crate) mod linear;
pub(crate) mod repeated_rnn;
pub(crate) mod sieve;
pub(crate) mod sieve_sep_center;

/// The algorithm to use for K-Nearest Neighbor search.
///
/// The default is `RepeatedRnn`, as determined by the benchmarks in the crate.
#[derive(Clone, Copy, Debug)]
pub enum Algorithm {
    /// Use linear search on the entire dataset.
    ///
    /// This is a stable algorithm.
    Linear,

    /// Use a repeated RNN search, increasing the radius until enough neighbors
    /// are found.
    ///
    /// This is a stable algorithm.
    ///
    /// Search starts with a radius equal to the radius of the tree divided by
    /// the cardinality of the dataset. If no neighbors are found, the radius is
    /// increased by a factor of 2 until at least one neighbor is found. Then,
    /// the radius is increased by a factor determined by the local fractal
    /// dimension of the neighbors found until enough neighbors are found. This
    /// factor is capped at 2. Once enough neighbors are found, the neighbors
    /// are sorted by distance and the first `k` neighbors are returned. Ties
    /// are broken arbitrarily.
    RepeatedRnn,

    /// Uses two priority queues and an increasing threshold to perform search.
    ///
    /// This algorithm is not stable.
    ///
    /// Begins with first priority queue, called `candidates`, wherein the top priority
    /// candidate is the one with the lowest d_min. We use d_min to express the
    /// theoretical closest a point in a given cluster can be to the query. Replaces
    /// the top priority candidate with its children until the top priority candidate
    /// is a leaf. Then, adds all instances in the leaf to a second priority queue, `hits`,
    /// wherein the top priority hit is the one with the highest distance to the query.
    /// Hits are then removed from the queue until the queue has size k. Repeats these steps
    /// until candidates is empty or the closest candidate is worse than the furthest hit.
    GreedySieve,
    /// Like SieveV1, but without the separate priority queue for hits.
    ///
    /// This algorithm is not stable.
    ///
    /// For each iteration of the search, we calculate a threshold from the
    /// `Cluster`s such that the k nearest neighbors of the query are guaranteed
    /// to be within the threshold. We then use this threshold to filter out
    /// clusters that are too far away from the query. Instead of maintaining
    /// a separate priority queue for hits, hits are treated as grains.
    ///
    /// This approach does not treat the center of a cluster separately from the rest
    /// of the points in the cluster.
    Sieve,

    /// Like SieveV2, but without the separate priority queue for hits.
    ///
    /// This algorithm is not stable.
    ///
    /// For each iteration of the search, we calculate a threshold from the
    /// `Cluster`s such that the k nearest neighbors of the query are guaranteed
    /// to be within the threshold. We then use this threshold to filter out
    /// clusters that are too far away from the query. Instead of maintaining
    /// a separate priority queue for hits, hits are treated as grains.
    ///
    /// This approach treats the center of a cluster separately from the rest
    /// of the points in the cluster.
    SieveSepCenter,
}

impl Default for Algorithm {
    fn default() -> Self {
        Self::RepeatedRnn
    }
}

impl Algorithm {
    /// Searches for the nearest neighbors of a query.
    ///
    /// # Arguments
    ///
    /// * `query` - The query to search around.
    /// * `k` - The number of neighbors to search for.
    /// * `tree` - The tree to search.
    ///
    /// # Returns
    ///
    /// A vector of 2-tuples, where the first element is the index of the instance
    /// and the second element is the distance from the query to the instance.
    pub(crate) fn search<I, U, D>(self, tree: &Tree<I, U, D>, query: &I, k: usize) -> Vec<(usize, U)>
    where
        I: Instance,
        U: Number,
        D: Dataset<I, U>,
    {
        match self {
            Self::Linear => {
                let indices = (0..tree.cardinality()).collect::<Vec<_>>();
                linear::search(tree.data(), query, k, &indices)
            }
            Self::RepeatedRnn => repeated_rnn::search(tree, query, k),
            Self::GreedySieve => greedy_sieve::search(tree, query, k),
            Self::Sieve => sieve::search(tree, query, k),
            Self::SieveSepCenter => sieve_sep_center::search(tree, query, k),
        }
    }

    /// Returns the name of the algorithm.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Linear => "Linear",
            Self::RepeatedRnn => "RepeatedRnn",
            Self::GreedySieve => "GreedySieve",
            Self::Sieve => "Sieve",
            Self::SieveSepCenter => "SieveSepCenter",
        }
    }

    /// Returns a list of all the algorithms, excluding Linear.
    #[must_use]
    pub const fn variants<'a>() -> &'a [Self] {
        &[Self::RepeatedRnn, Self::GreedySieve, Self::Sieve, Self::SieveSepCenter]
    }
}

/// A priority queue of hits for K-Nearest Neighbor search.
pub(crate) struct Hits<I: Hash + Eq + Copy, U: Number> {
    /// The priority queue of hits.
    pub queue: PriorityQueue<I, OrdNumber<U>>,
    /// The number of neighbors to search for.
    pub capacity: usize,
}

impl<I: Hash + Eq + Copy, U: Number> Hits<I, U> {
    /// Creates a new priority queue of hits.
    ///
    /// The priority queue is initialized with a `capacity` and is maintained
    /// at that `capacity`.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The number of neighbors to search for.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: PriorityQueue::with_capacity(capacity),
            capacity,
        }
    }

    /// Creates a new priority queue of hits from a vector of hits.
    pub fn from_vec(capacity: usize, vec: Vec<(I, U)>) -> Self {
        let mut queue = PriorityQueue::with_capacity(capacity);
        for (i, d) in vec {
            queue.push(i, OrdNumber(d));
        }
        while queue.len() > capacity {
            queue.pop();
        }
        Self { queue, capacity }
    }

    /// Number of hits in the queue.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Whether the queue is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the distance of the farthest hit in the queue.
    ///
    /// If the queue is empty, returns the result of calling `default`.
    pub fn peek(&self) -> U {
        self.queue.peek().map_or_else(U::zero, |(_, &OrdNumber(d))| d)
    }

    /// Pushes a hit onto the queue.
    ///
    /// If the queue is not full, the hit is pushed onto the queue. If the queue
    /// is full and the distance of the hit is less than the distance of the
    /// farthest hit in the queue, the farthest hit is popped from the queue and
    /// the new hit is pushed onto the queue.
    ///
    /// # Arguments
    ///
    /// * `i` - The index of the hit.
    /// * `d` - The distance of the hit.
    pub fn push(&mut self, i: I, d: U) {
        if self.queue.len() < self.capacity {
            self.queue.push(i, OrdNumber(d));
        } else if d < self.peek() {
            self.queue.pop();
            self.queue.push(i, OrdNumber(d));
        }
    }

    /// Push a batch of items onto the queue and reconcile with capacity at the
    /// end.
    pub fn push_batch(&mut self, items: impl Iterator<Item = (I, U)>) {
        items.for_each(|(i, d)| {
            self.queue.push(i, OrdNumber(d));
        });
        while self.queue.len() > self.capacity {
            self.queue.pop();
        }
    }

    /// Pops hits from the queue until the distance of the farthest hit is no
    /// farther than the given `threshold`.
    #[allow(dead_code)]
    pub fn pop_until(&mut self, threshold: U) {
        while threshold < self.peek() {
            self.queue.pop();
        }
    }

    /// Extracts the hits from the queue.
    pub fn extract(&self) -> Vec<(I, U)> {
        self.queue.iter().map(|(&i, &OrdNumber(d))| (i, d)).collect()
    }
}

/// Field by which we rank elements in priority queue of hits.
#[derive(Debug)]
pub struct OrdNumber<U: Number>(U);

impl<U: Number> PartialEq for OrdNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U: Number> Eq for OrdNumber<U> {}

impl<U: Number> PartialOrd for OrdNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for OrdNumber<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Greater)
    }
}

/// Field by which we reverse-rank elements in priority queue of hits.
#[derive(Debug)]
pub struct RevNumber<U: Number>(U);

impl<U: Number> PartialEq for RevNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<U: Number> Eq for RevNumber<U> {}

impl<U: Number> PartialOrd for RevNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for RevNumber<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Greater)
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use distances::Number;

    pub(crate) fn sort_hits<U: Number>(mut hits: Vec<(usize, U)>) -> Vec<(usize, U)> {
        hits.sort_by(|(_, i), (_, j)| i.partial_cmp(j).unwrap_or(Ordering::Greater));
        hits
    }
}
