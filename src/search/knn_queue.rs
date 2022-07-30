use std::cmp::Ordering;

use rand::Rng;

use crate::prelude::*;

#[derive(Debug)]
pub enum Delta {
    Delta0,
    Delta1,
    Delta2,
}

#[derive(Debug)]
pub struct KnnSieve<'a, T: Number, U: Number> {
    clusters: Vec<&'a Cluster<'a, T, U>>,
    query: &'a [T],
    k: usize,
}

impl<'a, T: Number, U: Number> KnnSieve<'a, T, U> {
    pub fn new(clusters: Vec<&'a Cluster<'a, T, U>>, query: &'a [T], k: usize) -> Self {
        KnnSieve { clusters, query, k }
    }

    fn compute_delta0(&self, c: &Cluster<T, U>) -> U {
        c.distance_to_instance(self.query)
    }

    fn compute_delta1(&self, c: &Cluster<T, U>, delta0: U) -> U {
        delta0 + c.radius()
    }

    fn compute_delta2(&self, c: &Cluster<T, U>, delta0: U) -> U {
        if delta0 > c.radius() {
            delta0 - c.radius()
        } else {
            U::zero()
        }
    }

    /// Gets the value of the given delta type (0, 1, or 2) to be used in knn search
    ///
    /// `delta0` = dist from query to cluster center
    /// `delta1` = dist from query to potentially farthest instance in cluster
    /// `delta2` = dist from query to potentially closest instance in cluster
    fn compute_delta(&self, c: &Cluster<T, U>, delta: &Delta) -> U {
        let delta0 = self.compute_delta0(c);
        match delta {
            Delta::Delta0 => delta0,
            Delta::Delta1 => self.compute_delta1(c, delta0),
            Delta::Delta2 => self.compute_delta2(c, delta0),
        }
    }

    pub fn sort_by_delta(mut self, delta: &Delta) -> Self {
        let deltas: Vec<_> = self.clusters.iter().map(|c| self.compute_delta(c, delta)).collect();
        let mut indices: Vec<_> = (0..deltas.len()).collect();
        indices.sort_by(|&i, &j| deltas[i].partial_cmp(&deltas[j]).unwrap());
        self.clusters = indices.into_iter().map(|i| self.clusters[i]).collect();
        self
    }

    /// Called by `random_partition()`. Takes the same arguments as `find_kth()` and returns the "true"
    /// index (i.e., index if `v` were sorted) of the Cluster currently at the rightmost possible index
    /// that could still have the `k`th Cluster (i.e.,`r`th index). Cluster at this `r`th index is based
    /// on random `pivot` from `random_partition()`.
    ///
    /// `j` tracks the index of the Cluster being evaluated at each iteration of the while loop. `i`
    /// counts the number of Clusters whose delta value is less than that of the `r`th Cluster.
    ///
    /// If a Cluster has a delta value greater than or equal to that of the `r`th Cluster, it  
    /// swaps position with the next Cluster whose delta is less than that of the `r`th Cluster.
    ///
    /// Since `i` counts the number of Clusters with a delta less than the `r`th Cluster, the final
    /// swap of the `i`th and `r`th Clusters puts that `r`th Cluster in its correct position (as if
    /// `v` were sorted).
    fn partition(&mut self, l: usize, r: usize, delta: &Delta) -> usize {
        let mut i = l;
        let mut j = l;

        while j < r {
            if self.compute_delta(self.clusters[j], delta) < self.compute_delta(self.clusters[r], delta) {
                self.clusters.swap(i, j);
                i += 1;
            }

            j += 1;
        }

        self.clusters.swap(i, r);
        i
    }

    /// Takes the same arguments as `find_kth()` and returns the "true" index (index if `v` were sorted)
    /// of the Cluster currently at a randomly selected `pivot` index.
    ///
    /// Chooses a random `pivot` value within the range plausible indices
    /// for the `k`th Cluster and swaps the `r`th Cluster with the Cluster in the pivot position.
    /// Calls `partition()` which evaluates Clusters in `v` against the `r`th Cluster (which
    /// is the Cluster formerly at the `pivot` index).
    ///
    /// `partition()` returns a count of Clusters with a delta value less than that at the `r`th
    /// (formerly, `pivot`) index (i.e., the index of `r`th Cluster if `v` were sorted); `random_partition()`
    /// returns this same value
    fn random_partition(&mut self, l: usize, r: usize, delta: &Delta) -> usize {
        let pivot = rand::thread_rng().gen_range(l..=r);
        self.clusters.swap(pivot, r);
        self.partition(l, r, delta)
    }

    /// `find_kth()` finds the kth Cluster in vector without completely sorting it
    /// Based on the QuickSelect Algorithm found here: https://www.geeksforgeeks.org/quickselect-algorithm/
    ///
    /// Takes mutable reference to a vector of Clusters (`v`), the leftmost index in `v` that could
    /// possibly have the `k`th cluster (`l`), the rightmost index in `v` that could possibly
    /// have the `k`th cluster (`r`), the desired index (`k`), and the desired delta type of the Cluster in the `k`th
    /// position if `v` were sorted (`delta`).
    ///
    /// With each recursive call, a random pivot is chosen and `partition_index` reflects the number of
    /// Clusters in `v` with a delta value less than that of the Cluster at the random pivot index. If
    /// `partition_index < k`, `find_kth()` calls itself with `l` adjusted to reflect the new possible indices
    /// for the `k`th Cluster (any index greater than `partition_index`). If `partition_index < k` `find_kth()` calls
    /// itself with `r` similarly adjusted.
    ///
    /// Recursion ceases when 'partition_index == k', at which point the desired delta value of the `k`th Cluster
    /// (`kth_delta`) is returned.
    pub fn find_kth(&mut self, delta: &Delta) -> U {
        self._find_kth(0, self.clusters.len() - 1, delta)
    }

    fn _find_kth(&mut self, l: usize, r: usize, delta: &Delta) -> U {
        let partition_index = self.random_partition(l, r, delta);

        match partition_index.cmp(&self.k) {
            Ordering::Less => self._find_kth(partition_index + 1, r, delta),
            Ordering::Greater => self._find_kth(l, partition_index - 1, delta),
            Ordering::Equal => self.compute_delta(self.clusters[self.k], delta),
        }
    }
}

// #[derive(Clone, Copy)]
// pub struct OrderedNumber<U: Number>(U);

// impl<U: Number> PartialEq for OrderedNumber<U> {
//     fn eq(&self, other: &Self) -> bool {
//         self.0 == other.0
//     }
// }

// impl<U: Number> Eq for OrderedNumber<U> {}

// impl<U: Number> PartialOrd for OrderedNumber<U> {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.0.partial_cmp(&other.0)
//     }
// }

// impl<U: Number> Ord for OrderedNumber<U> {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Greater)
//     }
// }

// type ClusterQueue<'a, T, U> = PriorityQueue<&'a Cluster<'a, T, U>, OrderedNumber<U>>;
// type InstanceQueue<'a, U> = PriorityQueue<usize, OrderedNumber<U>>;

// pub struct KnnQueue<'a, T: Number, U: Number> {
//     by_delta_0: ClusterQueue<'a, T, U>,
//     by_delta_1: ClusterQueue<'a, T, U>,
//     by_delta_2: ClusterQueue<'a, T, U>,
//     hits: InstanceQueue<'a, U>,
// }

// impl<'a, T: Number, U: Number> KnnQueue<'a, T, U> {
//     pub fn new(clusters_distances: &'a [(&Cluster<T, U>, U)]) -> Self {
//         let by_delta_0 = PriorityQueue::from_iter(clusters_distances.iter().map(|(c, d)| (*c, OrderedNumber(*d))));

//         let by_delta_1 = PriorityQueue::from_iter(
//             clusters_distances
//                 .iter()
//                 .map(|(c, d)| (*c, OrderedNumber(c.radius() + *d))),
//         );

//         let by_delta_2 = PriorityQueue::from_iter(clusters_distances.iter().map(|(c, d)| {
//             (
//                 *c,
//                 OrderedNumber(if c.radius() > *d { c.radius() - *d } else { U::zero() }),
//             )
//         }));

//         Self {
//             by_delta_0,
//             by_delta_1,
//             by_delta_2,
//             hits: PriorityQueue::new(),
//         }
//     }

//     pub fn by_delta_0(&self) -> Vec<&Cluster<T, U>> {
//         self.by_delta_0.clone().into_sorted_iter().map(|(c, _)| c).collect()
//     }

//     pub fn by_delta_1(&self) -> Vec<&Cluster<T, U>> {
//         self.by_delta_1.clone().into_sorted_iter().map(|(c, _)| c).collect()
//     }

//     pub fn by_delta_2(&self) -> Vec<&Cluster<T, U>> {
//         self.by_delta_2.clone().into_sorted_iter().map(|(c, _)| c).collect()
//     }

//     pub fn hits(&self) -> Vec<usize> {
//         self.hits.clone().into_sorted_vec()
//     }
// }

#[cfg(test)]
mod tests {

    use super::*;
    // use crate::prelude::*;

    #[test]
    fn find_2nd_delta0() {
        let data = vec![vec![0., 0., 0.], vec![1., 1., 1.], vec![2., 2., 2.], vec![3., 3., 3.]];
        let dataset = crate::Tabular::<f64>::new(&data, "test_cluster".to_string());
        let metric = metric_from_name::<f64, f64>("euclideansq", false).unwrap();
        let space = crate::TabularSpace::new(&dataset, metric.as_ref(), false);
        let partition_criteria = crate::PartitionCriteria::new(true)
            .with_max_depth(3)
            .with_min_cardinality(1);
        let cluster = Cluster::new_root(&space).build().partition(&partition_criteria, true);

        let flat_tree = cluster.subtree();
        let mut sieve = KnnSieve::new(flat_tree, &data[0], 1);

        // let mut v: Vec<DeltaValues<f64>> = vec![
        //     DeltaValues {
        //         delta0: 523.,
        //         delta1: 990.,
        //         delta2: 431.,
        //     },
        //     DeltaValues {
        //         delta0: 371.,
        //         delta1: 499.,
        //         delta2: 212.,
        //     },
        //     DeltaValues {
        //         delta0: 490.,
        //         delta1: 1097.,
        //         delta2: 117.,
        //     },
        //     DeltaValues {
        //         delta0: 242.,
        //         delta1: 947.,
        //         delta2: 198.,
        //     },
        //     DeltaValues {
        //         delta0: 761.,
        //         delta1: 866.,
        //         delta2: 514.,
        //     },
        //     DeltaValues {
        //         delta0: 241.,
        //         delta1: 281.,
        //         delta2: 131.,
        //     },
        //     DeltaValues {
        //         delta0: 520.,
        //         delta1: 824.,
        //         delta2: 378.,
        //     },
        // ];

        // let r = v.len() - 1;

        assert_eq!(sieve.find_kth(&Delta::Delta0), 3.);
    }
}
