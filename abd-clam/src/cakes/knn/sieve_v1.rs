//! Search function and helper functions for Knn thresholds approach with no separate grains for
//! cluster centers.

use core::cmp::{min, Ordering};

use distances::Number;

use crate::{Cluster, Dataset, Tree};

use super::Hits;

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

    let mut grains = vec![Grain::new(c, d)];
    let mut hits = Hits::new(k);

    loop {
        // The threshold is the minimum distance, so far, which guarantees that
        // the k nearest neighbors are within the threshold.
        let i = Grain::partition(&mut grains, k, hits.len());
        let threshold = grains[i].d;

        // Remove hits which are outside the threshold.
        hits.pop_until(threshold);

        let (insiders, non_insiders) = grains.split_at_mut(i);
        if non_insiders.is_empty() {
            grains_to_hits(data, query, &mut hits, insiders);
            break;
        }

        let (small_insiders, insiders) = insiders
            .iter_mut()
            .map(|&mut g| g)
            .partition::<Vec<_>, _>(|g| g.is_small(k));
        grains_to_hits(data, query, &mut hits, &small_insiders);

        let (leaves, non_leaves) = non_insiders
            .iter_mut()
            .filter(|g| !g.is_outside(threshold))
            .map(|&mut g| g)
            .partition::<Vec<_>, _>(|g| g.is_leaf);

        // If there are no non-leaves then the search is complete.
        if non_leaves.is_empty() {
            grains_to_hits(data, query, &mut hits, &insiders);
            grains_to_hits(data, query, &mut hits, &leaves);
            break;
        }

        grains = non_leaves
            .into_iter()
            .chain(insiders.into_iter())
            .flat_map(|g| {
                g.c.children()
                    .unwrap_or_else(|| unreachable!("This is only called on non-leaves."))
            })
            .map(|c| (c, c.distance_to_instance(data, query)))
            .map(|(c, d)| Grain::new(c, d))
            .chain(leaves.into_iter())
            .collect();
    }

    hits.extract()
}

/// Adds the instances in grains to the hits.
#[allow(clippy::needless_for_each)]
fn grains_to_hits<T, U, D>(data: &D, query: T, hits: &mut Hits<usize, U>, grains: &[Grain<T, U>])
where
    T: Send + Sync + Copy,
    U: Number,
    D: Dataset<T, U>,
{
    grains.iter().for_each(|g| {
        let indices = g.c.indices(data);
        let distances = data.query_to_many(query, indices);
        hits.push_batch(indices.iter().copied().zip(distances.into_iter()));
    });
}

/// A Grain is a structure which stores a cluster, a distance, and a multiplicity.
#[derive(Debug, Clone, Copy)]
struct Grain<'a, T: Send + Sync + Copy, U: Number> {
    /// The cluster.
    c: &'a Cluster<T, U>,
    /// The distance of the cluster's center to the query.
    d: U,
    /// The diameter of the cluster.
    diameter: U,
    /// The multiplicity of the cluster (in this version, multiplicity = cardinality)
    multiplicity: usize,
    /// Whether the cluster is a leaf.
    is_leaf: bool,
}

impl<'a, T: Send + Sync + Copy, U: Number> Grain<'a, T, U> {
    /// Creates a new instance of a Grain.
    fn new(c: &'a Cluster<T, U>, d: U) -> Self {
        let r = c.radius;
        Self {
            c,
            d: d + r,
            diameter: r + r,
            multiplicity: c.cardinality,
            is_leaf: c.is_leaf(),
        }
    }

    /// A `Grain` is small if its multiplicity is less than or equal to k or if
    /// its cluster is a leaf.
    pub const fn is_small(&self, k: usize) -> bool {
        (self.multiplicity <= k) || self.is_leaf
    }

    /// A Grain is "outside" the threshold if the closest, best-case possible point is further than
    /// the threshold distance to the query.
    pub fn is_outside(&self, threshold: U) -> bool {
        if self.d <= self.diameter {
            // query is inside the cluster
            false
        } else {
            threshold < (self.d - self.diameter)
        }
    }

    /// Wrapper function for `_partition_kth`.
    pub fn partition(grains: &mut [Self], k: usize, g_init: usize) -> usize {
        Self::_partition(grains, k, 0, grains.len() - 1, g_init)
    }

    /// Finds the smallest index i such that all grains with distance closer to or
    /// equal to the distance of the grain at index i have a multiplicity greater
    /// than or equal to k.
    #[allow(clippy::many_single_char_names)]
    fn _partition(grains: &mut [Self], k: usize, l: usize, r: usize, g_init: usize) -> usize {
        if l >= r {
            min(l, r)
        } else {
            let p = Self::_partition_once(grains, l, r);

            // The number of guaranteed hits within the first p grains.
            let g: usize = grains.iter().take(p).map(|g| g.multiplicity).sum();

            match g.cmp(&(k - g_init)) {
                Ordering::Equal => p,
                Ordering::Less => Self::_partition(grains, k, p + 1, r, g_init),
                Ordering::Greater => {
                    if (p > 0) && (g > (k + grains[p - 1].multiplicity)) {
                        Self::_partition(grains, k, l, p - 1, g_init)
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
    fn _partition_once(grains: &mut [Self], l: usize, r: usize) -> usize {
        let pivot = l + (r - l) / 2;
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

#[cfg(test)]
mod tests {

    use distances::vectors::euclidean;
    use symagen::random_data;

    use crate::{cakes::knn::linear, knn::tests::sort_hits, Cakes, PartitionCriteria, VecDataset};

    #[test]
    fn sieve_v1() {
        let (cardinality, dimensionality) = (1_000, 10);
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

            assert_eq!(linear_nn, sieve_nn);
        }
    }
}
