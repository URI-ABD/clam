use std::cmp::Ordering;

use crate::prelude::*;

#[derive(Debug)]
pub enum Delta {
    Delta0,
    Delta1,
    Delta2,
}

#[derive(Debug)]
//See note above sieve_swap about choice to have 3 separate delta members

/// Struct for facilitating knn tree search
/// `clusters` represents the list of candidate Clusters
/// The ith element of `cumulative_cardinalities` represents the sum of cardinalities of the 0th through ith Cluster in
/// `clusters`.
pub struct KnnSieve<'a, T: Number, U: Number> {
    pub clusters: Vec<&'a Cluster<'a, T, U>>,
    query: &'a [T],
    k: usize,
    pub cumulative_cardinalities: Vec<usize>,
    pub deltas_0: Vec<U>,
    deltas_1: Vec<U>,
    deltas_2: Vec<U>,
}

impl<'a, T: Number, U: Number> KnnSieve<'a, T, U> {
    pub fn new(clusters: Vec<&'a Cluster<'a, T, U>>, query: &'a [T], k: usize) -> Self {
        KnnSieve {
            clusters,
            query,
            k,
            cumulative_cardinalities: vec![],
            deltas_0: vec![],
            deltas_1: vec![],
            deltas_2: vec![],
        }
    }

    pub fn build(mut self) -> Self {
        self.cumulative_cardinalities = self.clusters
            .iter()
            .scan(0, |acc, c| {
                *acc += c.cardinality();
                Some(*acc)
            })
            .collect();
        
        self.deltas_0 = self.clusters
            .iter()
            .map(|&c| self.d0(c))
            .collect();
        
        (self.deltas_1, self.deltas_2) = self.clusters
            .iter()
            .zip(self.deltas_0.iter())
            .map(|(&c, d0)| (self.d1(c, *d0), self.d2(c, *d0)))
            .unzip();

        self
    }

    fn d0(&self, c: &Cluster<T, U>) -> U {
        c.distance_to_instance(self.query)
    }

    fn d1(&self, c: &Cluster<T, U>, d0: U) -> U {
        d0 + c.radius()
    }

    fn d2(&self, c: &Cluster<T, U>, d0: U) -> U {
        if d0 > c.radius() {
            d0 - c.radius()
        } else {
            U::zero()
        }
    }

    /// `sort_by_delta_0`, `sort_by_delta_1`, `sort_by_delta_2` sort `clusters` by the desired delta value for each
    /// cluster.
    /// NOTE: Right now, these do not also update the delta lists to reflect the sorted order of `clusters`, so
    /// after the sort, the delta lists will be outdated. This will cause issues. I don't think need to update all of them--
    /// only those deltas which will be a sort criterion in the subsequent step. After the sorting step is the replacement
    /// (with child Clusters) step and after that all of the deltas will need to be recomputed anyway.
    // pub fn sort_by_delta_0(mut self) -> Self {
    //     let mut indices: Vec<_> = (0..self.deltas_0.len()).collect();
    //     indices.sort_by(|&i, &j| self.deltas_0[i].partial_cmp(&self.deltas_0[j]).unwrap());
    //     self.clusters = indices.into_iter().map(|i| self.clusters[i]).collect();
    //     self
    // }

    fn update_deltas(&mut self, indices: &[usize]) {
        self.deltas_0 = indices.iter().map(|i| self.deltas_0[*i]).collect();
        self.deltas_1 = indices.iter().map(|i| self.deltas_1[*i]).collect();
        self.deltas_2 = indices.iter().map(|i| self.deltas_2[*i]).collect();
    }

    pub fn sort_by_delta_0(&mut self) {
        let mut indices: Vec<_> = (0..self.deltas_0.len()).collect();
        indices.sort_by(|&i, &j| self.deltas_0[i].partial_cmp(&self.deltas_0[j]).unwrap());
        let deltas_indices = indices.clone();

        self.clusters = indices.into_iter().map(|i| self.clusters[i]).collect();
        self.update_deltas(&deltas_indices);
    }

    pub fn sort_by_delta_1(&mut self) {
        let mut indices: Vec<_> = (0..self.deltas_1.len()).collect();
        indices.sort_by(|&i, &j| self.deltas_1[i].partial_cmp(&self.deltas_1[j]).unwrap());
        let deltas_indices = indices.clone();

        self.clusters = indices.into_iter().map(|i| self.clusters[i]).collect();
        self.update_deltas(&deltas_indices);
    }

    pub fn sort_by_delta_2(&mut self) {
        let mut indices: Vec<_> = (0..self.deltas_2.len()).collect();
        indices.sort_by(|&i, &j| self.deltas_2[i].partial_cmp(&self.deltas_2[j]).unwrap());
        let deltas_indices = indices.clone();

        self.clusters = indices.into_iter().map(|i| self.clusters[i]).collect();
        self.update_deltas(&deltas_indices);
    }

    /// Public version of _find_kth()
    pub fn find_kth(&mut self, delta: &Delta) -> U {
        self._find_kth(0, self.clusters.len() - 1, delta)
    }

    // NOTE: I don't know if maintaining all three deltas lists is the best way to do things.
    //
    // It would be possible to just maintain deltas_0 and compute delta1 and delta2 for a particular cluster from deltas_0
    // without ever actually having (and thus without maintaining) deltas_1 and deltas_2. That seems like a lot of calcultion
    // though, especially when filtering and sorting will rely heavily on delta1 and delta2 values.
    //
    // Another option would be to maintain only deltas_0 during find_kth and then recompute lists of deltas_1 and deltas_2 only
    // before sorting and filtering stage

    /// Swaps elements at given indices, `a` and `b` in `clusters` and each list of deltas such that the ith element in each delta list
    /// is the delta value for the ith Cluster in `clusters`.
    fn sieve_swap(&mut self, a: usize, b: usize) {
        self.clusters.swap(a, b);
        self.deltas_0.swap(a, b);
        self.deltas_1.swap(a, b);
        self.deltas_2.swap(a, b);
    }

    /// Updates `cumulative_cardinalities` to reflect changes in the order of `clusters.` `l` and `r` specify
    /// first and last indices in `cumulative_cardinalities` to edit respectively
    ///
    /// Requires l >= 0, r < self.clusters.len()
    fn update_cumulative_cardinalities(&mut self, l: usize, r: usize) {
        let range = if l == 0 {
            self.cumulative_cardinalities[0] = self.clusters[0].cardinality();
            1..=r
        } else {
            l..=r
        };
        for (i, &c) in range.zip(self.clusters.iter().skip(l)) {
            self.cumulative_cardinalities[i] = self.cumulative_cardinalities[i - 1] + c.cardinality();
        }
    }

    /// Computes `delta0` = dist from query to cluster center
    fn compute_delta0(c: &Cluster<T, U>, query: &[T]) -> U {
        c.distance_to_instance(query)
    }

    /// Computes `delta1` = dist from query to potentially farthest instance in cluster
    fn compute_delta1(c: &Cluster<T, U>, delta0: U) -> U {
        delta0 + c.radius()
    }

    /// Computes `delta2` = dist from query to potentially closest instance in cluster
    fn compute_delta2(c: &Cluster<T, U>, delta0: U) -> U {
        if delta0 > c.radius() {
            delta0 - c.radius()
        } else {
            U::zero()
        }
    }

    /// Currently, get_delta_by_cluster_index is being used where this was
    /// Gets the value of the given delta type (0, 1, or 2) based on cluster
    ///
    /// `delta0` = dist from query to cluster center
    /// `delta1` = dist from query to potentially farthest instance in cluster
    /// `delta2` = dist from query to potentially closest instance in cluster
    fn compute_delta(c: &Cluster<T, U>, query: &[T], delta: &Delta) -> U {
        let delta0 = KnnSieve::compute_delta0(c, query);
        match delta {
            Delta::Delta0 => delta0,
            Delta::Delta1 => KnnSieve::compute_delta1(c, delta0),
            Delta::Delta2 => KnnSieve::compute_delta2(c, delta0),
        }
    }

    /// Gets the value of the given delta type (0, 1, or 2) based on index (of Cluster)
    ///
    /// `delta0` = dist from query to Cluster center
    /// `delta1` = dist from query to potentially farthest instance in Cluster
    /// `delta2` = dist from query to potentially closest instance in Cluster
    fn get_delta_by_cluster_index(&self, index: usize, delta: &Delta) -> U {
        match delta {
            Delta::Delta0 => self.deltas_0[index],
            Delta::Delta1 => self.deltas_1[index],
            Delta::Delta2 => self.deltas_2[index],
        }
    }

    /// Called by `find_kth()`. Takes the same arguments as `find_kth()` and returns the "true"
    /// index (i.e., index if `clusters` were fully sorted by `delta`) of the Cluster currently at the
    /// rightmost possible index that could still have the `k`th Cluster (i.e.,`r`th index).
    /// Cluster at this `r`th index is based on `pivot`, which is simply the element at floor(`r+l`/`2`).
    ///
    /// `j` tracks the index of the Cluster being evaluated at each iteration of the while loop. `i`
    /// counts the number of Clusters whose delta value is less than that of the `r`th Cluster.
    ///
    /// If a Cluster has a delta value greater than or equal to that of the `r`th Cluster, it  
    /// swaps position with the next Cluster whose delta is less than that of the `r`th Cluster.
    ///
    /// Since `i` counts the number of Clusters with a delta less than the `r`th Cluster, the final
    /// swap of the `i`th and `r`th Clusters puts that `r`th Cluster in its correct position (as if
    /// `i` were sorted).
    ///
    /// In `clusters`, each Cluster is of multiplicity 1. `k`, however, reflects the desired index in
    /// a list of Clusters where each Clusters' multiplicity is its cardinality. As a result, we maintain
    /// a list of cumulative sums of Cluster cardinalities, hence the call to `update_cumulative_cardinalities`
    fn partition(&mut self, l: usize, r: usize, delta: &Delta) -> usize {
        let pivot = (r + l) / 2;
        self.sieve_swap(pivot, r);

        let mut i = l;
        let mut j = l;

        while j < r {
            if self.get_delta_by_cluster_index(j, delta) < self.get_delta_by_cluster_index(r, delta) {
                self.sieve_swap(i, j);
                i += 1;
            }

            j += 1;
        }

        self.sieve_swap(i, r);
        self.update_cumulative_cardinalities(l, r);

        i
    }

    /// Finds the `k`th Cluster in a theoretical list containing the Clusters in `clusters` where
    /// each Clusters' multiplicity is its cardinality
    /// Based on the QuickSelect Algorithm found here: https://www.geeksforgeeks.org/quickselect-algorithm/
    ///
    /// Takes `clusters` (candidate list of Clusters), the leftmost index in `clusters` that could
    /// possibly have the `k`th cluster (`l`), the rightmost index in `clusters` that could possibly
    /// have the `k`th cluster (`r`), the desired index (`k`), and the desired delta type of the Cluster in the `k`th
    /// position if `v` were sorted (`delta`). Note that while `k` reflects an index in a theoretical list,
    /// `l` and `r` are actual, legitimate indices of `clusters`.
    ///
    /// With each recursive call, a `pivot` index is chosen and `partition_index` reflects the number of
    /// Clusters in `clusters` with a delta value less than that of the Cluster at the `pivot` index.
    ///
    /// If `cumulative_cardinalities`[`partition_index`] < `k`, partition index is too low, and `find_kth()`
    /// calls itself with `l` adjusted to reflect the new possible indices for the `k`th Cluster.
    ///
    /// If `cumulative_cardinalities`[`partition_index`] == `k`, recursion ceases and the desired delta value of
    /// the `k`th Cluster is returned.
    ///
    /// If `cumulative_cardinalities`[partition_index] > `k`, we must also consider
    /// cumulative_cardinalities`[`partition_index`-1]. If `cumulative_cardinalities`[`partition_index`-1] < `k`, then
    /// we can infer that the `k`th Cluster is one of the theoretical copies of the Cluster at `partition_index`, in which
    /// case recursion ceases and the desired delta value of the `k`th Cluster is returned. Otherwise, the returned
    /// `partition_index` is too high, and `_find_kth()` calls itself with `r` adjusted to reflect the new possible
    /// indices for the `k`th Cluster.
    fn _find_kth(&mut self, l: usize, r: usize, delta: &Delta) -> U {
        let partition_index = self.partition(l, r, delta);

        match self.cumulative_cardinalities[partition_index].cmp(&self.k) {
            Ordering::Less => self._find_kth(partition_index + 1, r, delta),
            Ordering::Equal => self.get_delta_by_cluster_index(partition_index, delta), //terminate search immediately and take all clusters from 0 to i
            Ordering::Greater => {
                if self.cumulative_cardinalities[partition_index - 1] > self.k {
                    self._find_kth(l, partition_index - 1, delta)
                } else {
                    self.get_delta_by_cluster_index(partition_index, delta)
                }
            }
        }
    }

    pub fn replace_with_child_clusters(mut self) -> Self {
        (self.clusters, self.deltas_0, self.deltas_1, self.deltas_2) = self
            .clusters
            .into_iter()
            .zip(self.deltas_0.into_iter())
            .zip(self.deltas_1.into_iter())
            .zip(self.deltas_2.into_iter())
            .fold(
                (vec![], vec![], vec![], vec![]),
                |(mut c, mut d0, mut d1, mut d2), (((c_, d0_), d1_), d2_)| {
                    if c_.is_leaf() {
                        c.push(c_);
                        d0.push(d0_);
                        d1.push(d1_);
                        d2.push(d2_);
                    } else {
                        let [l, r] = c_.children();
                        c.extend_from_slice(&[l, r]);
                        let l0 = Self::compute_delta0(l, self.query);
                        let r0 = Self::compute_delta0(r, self.query);
                        d0.extend_from_slice(&[l0, r0]);
                        d1.extend_from_slice(&[Self::compute_delta1(l, l0), Self::compute_delta1(r, r0)]);
                        d2.extend_from_slice(&[Self::compute_delta2(l, l0), Self::compute_delta2(r, r0)]);
                    }
                    (c, d0, d1, d2)
                },
            );

        self
    }

    pub fn filter(mut self) -> Self {
        let d1_k = self.find_kth(&Delta::Delta1);
        let keep = self.deltas_2.iter().map(|d2| *d2 <= d1_k); 

        (self.clusters, self.deltas_0, self.deltas_1, self.deltas_2) = self
            .clusters
            .into_iter()
            .zip(self.deltas_0.into_iter())
            .zip(self.deltas_1.into_iter())
            .zip(self.deltas_2.clone().into_iter())
            .zip(keep)
            .fold(
                (vec![], vec![], vec![], vec![]), 
                |(mut c, mut d0, mut d1, mut d2), ((((c_, d0_), d1_), d2_), k_) | {
                    if k_ {
                        c.push(c_);
                        d0.push(d0_);
                        d1.push(d1_);
                        d2.push(d2_);
                    }
                    (c, d0, d1, d2)

                }, 
            ); 

        self.update_cumulative_cardinalities(0, self.clusters.len() - 1);
        self
    }

    pub fn are_all_leaves(&self) -> bool {
        self.clusters.iter().all(|c| c.is_leaf())
    }

    /// Returns `k` best hits from the sieve along with their distances from the
    /// query. If this method is called when the `are_all_leaves` method
    /// evaluates to `true`, the result will have the best recall. If the
    /// `metric` in use obeys the triangle inequality, then the results will
    /// have perfect recall. If this method is called before the sieve has been
    /// filtered down to the leaves, the results may not have perfect recall.
    pub fn extract(&self) -> Vec<(usize, U)> {
        todo!()
    }

    // fn filter(mut self, kth_delta1: U) -> Self{
    //     self.clusters = self.clusters.into_iter().enumerate().filter(|(i, v)| self.deltas_0[*i] > kth_delta1).collect();

    //     self
    // // }

    // fn knn_search(mut self) {
    //     while self.clusters.len() < self.k {
    //         self.replace_with_child_clusters()
    //     }

    //     let kth_delta1 = self.find_kth(&Delta::Delta1);
    //     //filter clusters whose delta 2 greater than kth delta 1
    //     self.filter();
    // }
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
