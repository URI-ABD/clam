//! Search function and helper functions for Knn thresholds approach with no separate grains for
//! cluster centers.

use core::cmp::{min, Ordering};

use distances::Number;

use crate::{Cluster, Dataset, Tree};

#[derive(Clone, Copy, Debug)]

/// A Grain is an element of the sieve. It is either a hit or a cluster.
enum Grain<'a, T: Send + Sync + Copy, U: Number> {
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
        c: &'a Cluster<T, U>,
        /// Theoretical worst case distance from the query to a point in the cluster.
        d: U,
        /// The diameter of the cluster.
        diameter: U,
        /// The number of instances in the cluster.
        multiplicity: usize,
        /// Whether the cluster is a leaf.
        is_leaf: bool,
    },
}

impl<'a, T: Send + Sync + Copy, U: Number> Grain<'a, T, U> {
    /// Creates a new `Grain` from a cluster.
    fn new_cluster(c: &'a Cluster<T, U>, d: U) -> Self {
        let r = c.radius;
        Self::Cluster {
            c,
            d: d + r,
            diameter: r + r,
            multiplicity: c.cardinality,
            is_leaf: c.is_leaf(),
        }
    }

    /// Creates a new `Grain` from a hit.
    const fn new_hit(d: U, index: usize) -> Self {
        Self::Hit { d, index }
    }

    /// Returns the theoretical minimum distance from the query to a point in
    /// the cluster if the `Grain` is of the `Cluster` variant; returns the
    /// distance to the instance if the `Grain` is of the `Hit` variant.
    fn d_min(&self) -> U {
        match self {
            Grain::Hit { d, .. } => *d,
            Grain::Cluster { d, diameter, .. } => *d - *diameter,
        }
    }

    /// Returns the theoretical maximum distance from the query to a point in
    /// the cluster if the `Grain` is of the `Cluster` variant; returns the
    /// distance to the instance if the `Grain` is of the `Hit` variant.
    const fn d(&self) -> U {
        match self {
            Grain::Hit { d, .. } | Grain::Cluster { d, .. } => *d,
        }
    }

    /// Returns whether the `Grain` has k or fewer instances or is a leaf.
    const fn is_small(&self, k: usize) -> bool {
        match self {
            Grain::Hit { .. } => true,
            Grain::Cluster {
                multiplicity, is_leaf, ..
            } => *multiplicity <= k || *is_leaf,
        }
    }

    /// Returns whether the `Grain` is a cluster.
    const fn is_cluster(&self) -> bool {
        match self {
            Grain::Hit { .. } => false,
            Grain::Cluster { .. } => true,
        }
    }

    /// Returns whether the `Grain` is outside the threshold.
    fn is_outside(&self, threshold: U) -> bool {
        self.d_min() > threshold
    }

    /// Returns the indices of the instances in the cluster if the `Grain` is of
    /// the `Cluster` variant
    fn cluster_to_hits(self, data: &impl Dataset<T, U>, query: T) -> Vec<Grain<T, U>> {
        match self {
            Grain::Hit { .. } => unreachable!("This is only called on non-hits."),
            Grain::Cluster { c, .. } => {
                let indices = c.indices(data);
                let distances = data.query_to_many(query, indices);
                indices
                    .iter()
                    .copied()
                    .zip(distances.into_iter())
                    .map(|(index, d)| Grain::new_hit(d, index))
                    .collect::<Vec<_>>()
            }
        }
    }

    /// Returns the children of the cluster if the `Grain` is of the `Cluster`
    fn cluster_to_children(self) -> [&'a Cluster<T, U>; 2] {
        match self {
            Grain::Hit { .. } => unreachable!("This is only called on non-hits."),
            Grain::Cluster { c, .. } => c
                .children()
                .unwrap_or_else(|| unreachable!("This is only called on non-leaves.")),
        }
    }

    /// Returns the multiplicity of the `Grain`.
    const fn multiplicity(&self) -> usize {
        match self {
            Grain::Hit { .. } => 1,
            Grain::Cluster { multiplicity, .. } => *multiplicity,
        }
    }

    /// Wrapper function for `_partition_kth`.
    pub fn partition(grains: &mut [Self], k: usize) -> usize {
        Self::_partition(grains, k, 0, grains.len() - 1)
    }

    /// Changes pivot point and swaps elements around so that all elements to left
    /// of pivot are less than or equal to pivot and all elements to right of pivot
    /// are greater than pivot.
    fn _partition_once(grains: &mut [Self], l: usize, r: usize) -> usize {
        let pivot = l + (r - l) / 2;
        grains.swap(pivot, r);

        let (mut a, mut b) = (l, l);
        while b < r {
            if grains[b].d() < grains[r].d() {
                grains.swap(a, b);
                a += 1;
            }
            b += 1;
        }

        grains.swap(a, r);

        a
    }

    /// Finds the smallest index i such that all grains with distance closer to or
    /// equal to the distance of the grain at index i have a multiplicity greater
    /// than or equal to k.
    #[allow(clippy::many_single_char_names)]
    fn _partition(grains: &mut [Self], k: usize, l: usize, r: usize) -> usize {
        if l >= r {
            min(l, r)
        } else {
            let p = Self::_partition_once(grains, l, r);

            // The number of guaranteed hits within the first p grains.
            let g = grains.iter().take(p).map(Grain::multiplicity).sum::<usize>();
            match g.cmp(&k) {
                Ordering::Equal => p - 1,
                Ordering::Less => Self::_partition(grains, k, p + 1, r),
                Ordering::Greater => {
                    if (p > 0) && (g > (k + grains[p - 1].multiplicity())) {
                        Self::_partition(grains, k, l, p - 1)
                    } else {
                        p - 1
                    }
                }
            }
        }
    }

    /// Returns the index of the instance if the `Grain` is of the `Hit` variant.
    fn index(&self) -> usize {
        match self {
            Grain::Hit { index, .. } => *index,
            Grain::Cluster { .. } => unreachable!("This is only called on hits."),
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
pub fn search<T, U, D>(tree: &Tree<T, U, D>, query: T, k: usize) -> Vec<(usize, U)>
where
    T: Send + Sync + Copy,
    U: Number,
    D: Dataset<T, U>,
{
    let data = tree.data();
    let c = tree.root();
    let d = c.distance_to_instance(data, query);

    let mut grains = vec![Grain::new_cluster(c, d)];

    loop {
        // The threshold is the minimum distance, so far, which guarantees that
        // the k nearest neighbors are within the threshold.
        let i = Grain::partition(&mut grains, k);
        let threshold = grains[i].d();

        // Remove grains which are outside the threshold.

        let (insiders, non_insiders) = grains.split_at_mut(i + 1);
        let non_insiders = non_insiders
            .iter_mut()
            .map(|&mut g| g)
            .filter(|g| !g.is_outside(threshold))
            .collect::<Vec<_>>();

        let (clusters, mut hits) = insiders
            .iter_mut()
            .map(|&mut g| g)
            .chain(non_insiders)
            .partition::<Vec<_>, _>(Grain::is_cluster);

        let (small_clusters, clusters) = clusters.into_iter().partition::<Vec<_>, _>(|g| g.is_small(k));

        for cluster in small_clusters {
            let new_hits = cluster.cluster_to_hits(data, query);
            hits.extend(new_hits.into_iter());
        }

        if clusters.is_empty() {
            let i = Grain::partition(&mut hits, k);
            return hits[0..=i].iter().map(|g| (g.index(), g.d())).collect();
        }

        grains = clusters
            .into_iter()
            .flat_map(Grain::cluster_to_children)
            .map(|c| Grain::new_cluster(c, c.distance_to_instance(data, query)))
            .chain(hits.into_iter())
            .collect();
    }
}

#[cfg(test)]
mod tests {

    use core::f32::EPSILON;

    use distances::vectors::euclidean;
    use symagen::random_data;

    use crate::{cakes::knn::linear, knn::tests::sort_hits, Cakes, PartitionCriteria, VecDataset};

    #[test]
    fn sieve_v1() {
        let (cardinality, dimensionality) = (10_000, 10);
        let (min_val, max_val) = (-1.0, 1.0);
        let seed = 42;

        let data = random_data::random_f32(cardinality, dimensionality, min_val, max_val, seed);
        let data = data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let data = VecDataset::new("knn-test".to_string(), data, euclidean::<_, f32>, false);

        let query = random_data::random_f32(1, dimensionality, min_val, max_val, seed * 2);
        let query = query[0].as_slice();

        let criteria = PartitionCriteria::default();
        let model = Cakes::new(data, Some(seed), criteria);
        let tree = model.tree();

        for k in [100, 10, 1] {
            let linear_nn = sort_hits(linear::search(tree.data(), query, k, tree.indices()));
            let sieve_nn = sort_hits(super::search(tree, query, k));

            assert_eq!(sieve_nn.len(), k);

            let d_linear = linear_nn[k - 1].1;
            let d_sieve = sieve_nn[k - 1].1;
            assert!(
                (d_linear - d_sieve).abs() < EPSILON,
                "k = {}, linear = {}, sieve = {}",
                k,
                d_linear,
                d_sieve
            );
        }
    }
}
