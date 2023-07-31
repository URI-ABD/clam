//! Helper functions for Knn thresholds approach with no separate grains for
//! cluster centers.

use distances::Number;
use priority_queue::DoublePriorityQueue;
use std::marker::PhantomData;

use std::cmp::Ordering;

use crate::core::cluster::{Cluster, Tree};
use crate::core::dataset::Dataset;

#[allow(dead_code)]

/// A `KnnSieve` is a data structure that is used to find the `k` nearest neighbors of a `query` point in a dataset.
///
/// The `KnnSieve` is initialized with a `tree`, a `query` point, and a `k` value. `tree` contains a
/// hierarchical clustering of the dataset. The `query`  is the point for which we want to find the `k` nearest
/// neighbors.
///
/// `grains` is a running list of `Grain`s (each grain consists of a cluster, a distance, and a multiplicity)
/// which could still contain one of the `k` nearest neighbors. `hits` is a priority queue of points which could
/// be one of the `k` nearest neighbors. `is refined` is a boolean which is true if hits contains exactly `k` points
/// and there are no more `Grain`s which can be partitioned.
pub struct KnnSieve<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> {
    /// The cluster tree to search.
    tree: &'a Tree<T, U, D>,
    /// The query point.
    query: T,
    /// The number of neighbors to find. The algorithm requires k > 0.
    k: usize,
    /// A vector of `Grains` which could still contain one of the k-nearest
    /// neighbors. A `Grain` is a cluster, a distance, and a multiplicity.
    grains: Vec<Grain<'a, T, U>>,
    /// Whether hits contains the k-nearest neighbors.
    is_refined: bool,
    /// A priority queue of points which could be a nearest neighbor.
    hits: priority_queue::DoublePriorityQueue<usize, OrdNumber<U>>,
}

impl<'a, T: Send + Sync + Copy, U: Number, D: Dataset<T, U>> KnnSieve<'a, T, U, D> {
    /// Creates a new instance of a `KnnSieve`.
    pub fn new(tree: &'a Tree<T, U, D>, query: T, k: usize) -> Self {
        Self {
            tree,
            query,
            k,
            grains: Vec::new(),
            is_refined: false,
            hits: DoublePriorityQueue::default(),
        }
    }

    /// One-time computation of `grains.`
    pub fn initialize_grains(&mut self) {
        let layer = vec![self.tree.root()];

        let distances = layer
            .iter()
            .map(|c| c.distance_to_instance(self.tree.data(), self.query))
            .collect::<Vec<_>>();

        self.grains = layer
            .into_iter()
            .zip(distances.iter())
            .map(|(c, &d)| Grain::new(c, d, c.cardinality))
            .collect::<Vec<_>>();
    }

    /// Returns whether search is complete, i.e., whether hits
    /// contains the k-nearest neighbors.
    pub const fn is_refined(&self) -> bool {
        self.is_refined
    }

    /// One iteration of the refinement step.
    pub fn refine_step(&mut self) {
        // Determines the index of the grain such that if we only retain that grain
        // and every closer grain, we are guaranteed to have at least k points.
        // Threshold is the distance of the furthest point in the grain.
        let i = Grain::partition_kth(&mut self.grains, self.k);
        let ith_grain = &self.grains[i];
        let threshold = ith_grain.d + ith_grain.c.radius;

        // Filters hits by being outside the threshold.
        // Ties are added to hits together; we will never remove too many instances here
        // because our choice of threshold guarantees enough instances.
        while !self.hits.is_empty() && self.hits.peek_max().unwrap_or_else(|| unreachable!("The first clause in this conjunction ensures that hits is not empty. Its cardinality is always finite, so it must have a maximum element.")).1.number > threshold {
            self.hits.pop_max().unwrap_or_else(|| unreachable!("The first clause in this conjunction ensures that hits is not empty. Its cardinality is always finite, so it must have a maximum element."));
        }

        // Partition into insiders and straddlers
        // where we filter for grains being outside the threshold could be made more
        // efficient by leveraging the fact that partition already puts items on the correct
        // side of the threshold element
        #[allow(clippy::iter_with_drain)] // clippy is wrong in this case
        let (mut insiders, straddlers): (Vec<_>, Vec<_>) = self
            .grains
            .drain(..)
            .filter(|g| !Grain::is_outside(g, threshold))
            .partition(|g| Grain::is_inside(g, threshold));

        // Distinguish between those insiders we won't partition further and those we will
        // Add instances from insiders we won't further partition to hits
        let (small_insiders, big_insiders): (Vec<_>, Vec<_>) = insiders
            .into_iter()
            .partition(|g| (g.multiplicity <= self.k) || g.c.is_leaf());
        insiders = big_insiders;

        for g in small_insiders {
            let new_hits = g.c.indices(self.tree.data()).iter().map(|&i| {
                (
                    i,
                    OrdNumber {
                        number: self.tree.data().query_to_one(self.query, i),
                    },
                )
            });
            self.hits.extend(new_hits);
        }

        // Descend into straddlers
        // If there are no straddlers or all of the straddlers are leaves, then the grains in insiders and straddlers
        // are added to hits. If there are more than k hits, we repeatedly remove the furthest instance in hits until
        // there are either k hits left or more than k hits with some ties
        // If straddlers is not empty nor all leaves, partition non-leaves into children
        if straddlers.is_empty() || straddlers.iter().all(|g| g.c.is_leaf()) {
            insiders.into_iter().chain(straddlers.into_iter()).for_each(|g| {
                let new_hits =
                    g.c.indices(self.tree.data())
                        .iter()
                        .map(|&i| (i, self.tree.data().query_to_one(self.query, i)))
                        .map(|(i, d)| (i, OrdNumber { number: d }));

                self.hits.extend(new_hits);
            });

            if self.hits.len() > self.k {
                let mut potential_ties = vec![self.hits.pop_max().unwrap_or_else(|| unreachable!("Since we have k > 0 and the line above ensures hits.len() > k, hits is non-empty. Its cardinality is always finite, so it must have a maximum element."))];
                while self.hits.len() >= self.k {
                    let item = self.hits.pop_max().unwrap_or_else(|| unreachable!("Since we have k > 0 and the line above ensures hits.len() >= k, hits is non-empty. Its cardinality is always finite, so it must have a maximum element."));
                    if item.1.number <= potential_ties.last().unwrap_or_else(|| unreachable!("Potential ties starts non-empty (as has already been verified, and an element is added to it for each iteration of the loop, so it will always have at least one element.")).1.number {
                        potential_ties.clear();
                    }
                    potential_ties.push(item);
                }
                self.hits.extend(potential_ties.into_iter());
            }

            self.is_refined = true;
        } else {
            self.grains = insiders.into_iter().chain(straddlers.into_iter()).collect();
            let (leaves, non_leaves): (Vec<_>, Vec<_>) = self.grains.drain(..).partition(|g| g.c.is_leaf());

            let children = non_leaves
                .into_iter()
                .flat_map(|g| {
                    g.c.children()
                        .unwrap_or_else(|| unreachable!("This is only called on non-leaves."))
                })
                .map(|c| (c, c.distance_to_instance(self.tree.data(), self.query)))
                .map(|(c, d)| Grain::new(c, d, c.cardinality));

            self.grains = leaves.into_iter().chain(children).collect();
        }
    }
    /// Returns the indices and distances to the query of the k nearest neighbors of the query.
    pub fn extract(&self) -> Vec<(usize, U)> {
        self.hits.iter().map(|(i, d)| (*i, d.number)).collect()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
/// A Grain is a structure which stores a cluster, a distance, and a multiplicity.
struct Grain<'a, T: Send + Sync + Copy, U: Number> {
    /// Just something we have to do.
    t: std::marker::PhantomData<T>,
    /// The cluster.
    c: &'a Cluster<T, U>,
    /// The distance of the cluster's center to the query.
    d: U,
    /// The multiplicity of the cluster (in this version, multiplicity = cardinality)
    multiplicity: usize,
}

impl<'a, T: Send + Sync + Copy, U: Number> Grain<'a, T, U> {
    /// Creates a new instance of a Grain.
    fn new(c: &'a Cluster<T, U>, d: U, multiplicity: usize) -> Self {
        let t = PhantomData::default();
        Self { t, c, d, multiplicity }
    }

    /// A Grain is "inside" the threshold if the furthest, worst-case possible point is at most as far as
    /// threshold distance from the query.
    fn is_inside(&self, threshold: U) -> bool {
        let d_max = self.d + self.c.radius;
        d_max < threshold
    }

    /// A Grain is "outside" the threshold if the closest, best-case possible point is further than
    /// the threshold distance to the query.
    fn is_outside(&self, threshold: U) -> bool {
        let d_min = if self.d < self.c.radius {
            U::zero()
        } else {
            self.d - self.c.radius
        };
        d_min > threshold
    }

    /// Wrapper function for `_partition_kth`.
    fn partition_kth(grains: &mut [Self], k: usize) -> usize {
        let i = Self::_partition_kth(grains, k, 0, grains.len() - 1);
        let t = grains[i].d;

        let mut b = i;
        for a in (i + 1)..(grains.len()) {
            if grains[a].d == t {
                b += 1;
                grains.swap(a, b);
            }
        }

        b
    }

    /// Finds the smallest index i such that all grains with distance closer to or equal to the distance of the grain at index i
    /// have a multiplicity greater than or equal to k.
    fn _partition_kth(grains: &mut [Self], k: usize, l: usize, r: usize) -> usize {
        if l >= r {
            std::cmp::min(l, r)
        } else {
            let p = Self::_partition(grains, l, r);
            let guaranteed = grains
                .iter()
                .scan(0, |acc, g| {
                    *acc += g.multiplicity;
                    Some(*acc)
                })
                .collect::<Vec<_>>();

            let num_g = guaranteed[p];

            match num_g.cmp(&k) {
                std::cmp::Ordering::Less => Self::_partition_kth(grains, k, p + 1, r),
                std::cmp::Ordering::Equal => p,
                std::cmp::Ordering::Greater => {
                    if (p > 0) && (guaranteed[p - 1] > k) {
                        Self::_partition_kth(grains, k, l, p - 1)
                    } else {
                        p
                    }
                }
            }
        }
    }

    /// Changes pivot point and swaps elements around so that all
    /// elements to left of pivot are less than or equal to pivot and all elements to right of pivot are greater than pivot.
    fn _partition(grains: &mut [Self], l: usize, r: usize) -> usize {
        let pivot = (l + r) / 2;
        grains.swap(pivot, r);

        let (mut a, mut b) = (l, l);
        while b < r {
            if grains[b].d <= grains[r].d {
                grains.swap(a, b);
                a += 1;
            }
            b += 1;
        }

        grains.swap(a, r);

        a
    }
}

#[derive(Debug)]

/// Field by which we rank elements in priority queue of hits.
pub struct OrdNumber<U: Number> {
    /// The number we use to rank elements (distance to query).
    pub number: U,
}

impl<U: Number> PartialEq for OrdNumber<U> {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
    }
}

impl<U: Number> Eq for OrdNumber<U> {}

impl<U: Number> PartialOrd for OrdNumber<U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.number.partial_cmp(&other.number)
    }
}

impl<U: Number> Ord for OrdNumber<U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or_else(|| {
            unreachable!(
                "All hits are instances, and
        therefore each hit has a distance from the query. Since all hits' distances to the
        query will be represented by the same type, we can always compare them."
            )
        })
    }
}
