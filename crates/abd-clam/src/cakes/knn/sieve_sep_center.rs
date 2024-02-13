//! Search function and helper functions for Knn thresholds approach with no separate grains for
//! cluster centers.

use core::cmp::{min, Ordering};

use distances::Number;

use crate::{Cluster, Dataset, Instance, Tree};

/// A Grain is an element of the sieve. It is either a hit or a cluster.
#[derive(Debug)]
enum Grain<'a, U: Number, C: Cluster<U>> {
    /// A `Hit` is a single instance.
    Hit {
        /// Distance from the query to the instance.
        d: U,
        /// Index of the instance.
        index: usize,
    },

    /// A `Cluster`.
    Cluster {
        /// The cluster.
        c: &'a C,
        /// Theoretical worst case distance from the query to a point in the cluster.
        d_max: U,
        /// Theoretical best case distance from the query to a point in the cluster.
        d_min: U,
        /// The number of instances in the cluster minus 1 for the center.
        multiplicity: usize,
        /// Whether the cluster is a leaf.
        is_leaf: bool,
    },

    /// A `Center` is the center of a cluster.
    Center {
        /// Distance from the query to the center.
        d: U,
    },
}

impl<'a, U: Number, C: Cluster<U>> Grain<'a, U, C> {
    /// Creates a new `Grain` from a cluster.
    fn new_cluster(c: &'a C, d: U) -> Self {
        let r = c.radius();
        Self::Cluster {
            c,
            d_max: d + r,
            d_min: if d > r { d - r } else { U::zero() },
            multiplicity: c.cardinality() - 1,
            is_leaf: c.is_leaf(),
        }
    }

    /// Creates a new `Grain` from a hit.
    const fn new_hit(d: U, index: usize) -> Self {
        Self::Hit { d, index }
    }

    /// Creates a new `Grain` from a cluster center.
    const fn new_center(d: U) -> Self {
        Self::Center { d }
    }

    /// Creates center and cluster grains from a cluster.
    fn new_grains<I: Instance, D: Dataset<I, U>>(c: &'a C, data: &D, query: &I) -> Vec<Self> {
        if c.is_singleton() {
            let d = c.distance_to_instance(data, query);
            c.indices().map(|i| Self::new_hit(d, i)).collect()
        } else if c.is_leaf() {
            let distances = data.query_to_many(query, &c.indices().collect::<Vec<_>>());
            c.indices().zip(distances).map(|(i, d)| Self::new_hit(d, i)).collect()
        } else {
            let d = c.distance_to_instance(data, query);
            vec![Self::new_cluster(c, d), Self::new_center(d)]
        }
    }

    /// Returns the theoretical minimum distance from the query to a point in
    /// the cluster if the `Grain` is of the `Cluster` variant; returns the
    /// distance to the instance if the `Grain` is of the `Hit` variant.
    const fn d_min(&self) -> U {
        match self {
            Grain::Hit { d, .. } | Grain::Cluster { d_min: d, .. } | Grain::Center { d, .. } => *d,
        }
    }

    /// Returns the theoretical maximum distance from the query to a point in
    /// the cluster if the `Grain` is of the `Cluster` variant; returns the
    /// distance to the instance if the `Grain` is of the `Hit` variant.
    const fn d_max(&self) -> U {
        match self {
            Grain::Hit { d, .. } | Grain::Cluster { d_max: d, .. } | Grain::Center { d, .. } => *d,
        }
    }

    /// Returns whether the `Grain` has k or fewer instances or is a leaf.
    const fn is_small(&self, k: usize) -> bool {
        match self {
            Grain::Hit { .. } | Grain::Center { .. } => true,
            Grain::Cluster {
                multiplicity, is_leaf, ..
            } => *multiplicity <= k || *is_leaf,
        }
    }

    /// Returns whether the `Grain` is outside the threshold.
    fn not_outside(&self, threshold: U) -> bool {
        self.d_min() <= threshold
    }

    /// Returns the indices of the instances in the cluster if the `Grain` is of
    /// the `Cluster` variant
    fn cluster_to_hits<I: Instance, D: Dataset<I, U>>(self, data: &D, query: &I) -> Vec<Self> {
        match self {
            Grain::Hit { .. } | Grain::Center { .. } => unreachable!("This is only called on Clusters."),
            Grain::Cluster { c, d_max, .. } => {
                if c.is_singleton() {
                    let d = d_max - c.radius();
                    c.indices().map(|index| Grain::new_hit(d, index)).collect()
                } else {
                    let distances = data.query_to_many(query, &c.indices().collect::<Vec<_>>());
                    c.indices()
                        .zip(distances)
                        .map(|(index, d)| Grain::new_hit(d, index))
                        .collect()
                }
            }
        }
    }

    /// Returns the children of the cluster if the `Grain` is of the `Cluster`
    fn cluster_to_children(self) -> [&'a C; 2] {
        match self {
            Grain::Hit { .. } | Grain::Center { .. } => unreachable!("This is only called on Clusters."),
            Grain::Cluster { c, .. } => c
                .children()
                .unwrap_or_else(|| unreachable!("This is only called on non-leaves.")),
        }
    }

    /// Returns the multiplicity of the `Grain`.
    const fn multiplicity(&self) -> usize {
        match self {
            Grain::Hit { .. } | Grain::Center { .. } => 1,
            Grain::Cluster { multiplicity, .. } => *multiplicity,
        }
    }

    /// Wrapper function for `_partition_kth`.
    pub fn partition(grains: &mut [Self], k: usize) -> usize {
        Self::_partition(grains, k, 0, grains.len() - 1)
    }

    /// Finds the smallest index i such that all grains with distance closer to or
    /// equal to the distance of the grain at index i have a multiplicity greater
    /// than or equal to k.
    #[allow(clippy::many_single_char_names)]
    fn _partition(grains: &mut [Self], k: usize, l: usize, r: usize) -> usize {
        if l >= r {
            min(l, r)
        } else {
            // If the mean cardinality of the grains in question is greater than
            // k, the leftmost index should be the pivot.
            let mean_cardinality = grains[l..=r].iter().map(Grain::multiplicity).sum::<usize>() / (r - l + 1);
            let pivot = if mean_cardinality > k { 0 } else { (r - l) / 2 };
            let p = l + Self::partition_once(&mut grains[l..=r], pivot);

            // The number of guaranteed hits within the first p grains.
            let g = grains.iter().take(p + 1).map(Grain::multiplicity).sum::<usize>();

            match g.cmp(&k) {
                Ordering::Equal => p,
                Ordering::Less => Self::_partition(grains, k, p + 1, r),
                Ordering::Greater => {
                    if p > l {
                        match g.cmp(&(k + grains[p - 1].multiplicity())) {
                            Ordering::Equal => p - 1,
                            Ordering::Less => p,
                            Ordering::Greater => Self::_partition(grains, k, l, p - 1),
                        }
                    } else {
                        p
                    }
                }
            }
        }
    }

    /// Changes pivot point and swaps elements around so that all elements to left
    /// of pivot are less than or equal to pivot and all elements to right of pivot
    /// are greater than pivot.
    #[allow(clippy::many_single_char_names)]
    fn partition_once(grains: &mut [Self], pivot: usize) -> usize {
        // If the pivot is l, don't bother doing any additional swaps and just
        // exchange the 0th element for the minimum.
        if pivot == 0 {
            let min = grains
                .iter()
                .map(Grain::d_max)
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
                .unwrap_or_else(|| unreachable!("grains is non-empty"))
                .0;
            grains.swap(pivot, min);
            pivot
        } else {
            let r = grains.len() - 1;
            grains.swap(pivot, r);

            let (mut a, mut b) = (0, 0);
            while b < r {
                if grains[b].d_max() < grains[r].d_max() {
                    grains.swap(a, b);
                    a += 1;
                }
                b += 1;
            }

            grains.swap(a, r);

            a
        }
    }

    /// Returns the index of the instance if the `Grain` is of the `Hit` variant.
    fn index(&self) -> usize {
        match self {
            Grain::Hit { index, .. } => *index,
            Grain::Cluster { .. } | Grain::Center { .. } => unreachable!("This is only called on hits."),
        }
    }
}

/// K-Nearest Neighbor search using a thresholds approach with no separate centers.
///
/// # Arguments
///
/// * `tree` - The tree to search.
/// * `query` - The query to search around.
/// * `k` - The number of neighbors to search for.
///
/// # Returns
///
/// A vector of 2-tuples, where the first element is the index of the instance
/// and the second element is the distance from the query to the instance.
pub fn search<I, U, D, C>(tree: &Tree<I, U, D, C>, query: &I, k: usize) -> Vec<(usize, U)>
where
    I: Instance,
    U: Number,
    D: Dataset<I, U>,
    C: Cluster<U>,
{
    let data = tree.data();
    let mut grains = Grain::new_grains(&tree.root, data, query);
    let [mut insiders, mut non_insiders]: [Vec<_>; 2];

    loop {
        // The threshold is the minimum distance, so far, which guarantees that
        // the k nearest neighbors are within the threshold.
        let i = Grain::partition(&mut grains, k);
        let threshold = grains[i].d_max();

        // Remove grains which are outside the threshold.
        non_insiders = grains.split_off(i + 1);
        insiders = grains;
        let non_insiders = non_insiders.into_iter().filter(|g| g.not_outside(threshold));

        // Partition the grains into hits and clusters.
        let (clusters, mut hits) = insiders
            .into_iter()
            .chain(non_insiders)
            .filter(|g| !matches!(g, Grain::Center { .. }))
            .partition::<Vec<_>, _>(|g| matches!(g, Grain::Cluster { .. }));

        // Separate small (cardinality <=k or leaf) clusters from the rest.
        let (small_clusters, clusters) = clusters.into_iter().partition::<Vec<_>, _>(|g| g.is_small(k));

        // Convert small clusters to hits.
        for cluster in small_clusters {
            hits.append(&mut cluster.cluster_to_hits(data, query));
        }

        // If there are no more cluster grains, then the search is complete.
        if clusters.is_empty() {
            Grain::partition(&mut hits, k);

            // TODO: Fix the panic here.
            let l = core::cmp::min(k, hits.len());

            return hits[..l].iter().map(|g| (g.index(), g.d_max())).collect();
        }

        // Partition clusters into children and convert to grains.
        grains = clusters
            .into_iter()
            .flat_map(Grain::cluster_to_children)
            .flat_map(|c| Grain::new_grains(c, data, query))
            .chain(hits)
            .collect();
    }
}
