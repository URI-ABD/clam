//! K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.

use core::borrow::Borrow;
use std::cmp::Reverse;

use rayon::prelude::*;

use crate::{
    DistanceValue,
    cakes::{KnnRrnn, RnnChess, d_max, d_min},
    utils::SizedHeap,
};

use super::super::{Codec, Compressible, CompressiveSearch, PancakesTree};

impl<Id, I, T, A, M, C> CompressiveSearch<Id, I, T, A, M, C> for KnnRrnn
where
    I: Compressible,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
    C: Codec<I>,
{
    fn compressive_search<Query: Borrow<I>>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)> {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().compressive_search(tree, query);
        }

        let mut candidate_radii = SizedHeap::new(None);

        let d = tree.distance_to_uncompressed(query, 0);
        let root = tree.get_cluster_unchecked(0);
        let (car, radius) = (root.cardinality, root.radius);
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((1 + car / 2, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest_id = 0;
        while !tree.get_cluster_unchecked(latest_id).is_leaf() {
            // We have not yet reached a leaf cluster, so we need to explore the children of the current cluster.
            if let Some(child_center_ids) = tree.decompress_child_centers(latest_id) {
                let distances = child_center_ids
                    .into_iter()
                    .map(|cid| tree.items[cid].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (cid, d)))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."));

                if let Some((cid, d)) = distances.into_iter().min_by_key(|&(_, d)| crate::utils::MinItem((), d)) {
                    let c = tree.get_cluster_unchecked(cid);
                    let (car, radius) = (c.cardinality, c.radius);
                    candidate_radii.push((1, Reverse(d_min(radius, d))));
                    candidate_radii.push((1 + car / 2, Reverse(d)));
                    candidate_radii.push((car, Reverse(d_max(radius, d))));

                    latest_id = cid;
                }
            }
        }

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in arrange_candidate_radii(candidate_radii) {
            if e < self.k {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess::new(d).compressive_search(tree, query);
            if hits.len() >= self.k {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.k);
                break;
            }
        }
        hits
    }

    fn par_compressive_search<Query: Borrow<I> + Send + Sync>(&self, tree: &mut PancakesTree<Id, I, T, A, M, C>, query: &Query) -> Vec<(usize, T)>
    where
        Self: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        I::Compressed: Send + Sync,
        T: Send + Sync,
        A: Send + Sync,
        M: Send + Sync,
        C: Send + Sync,
    {
        if self.k > tree.cardinality() {
            // If k is greater than the number of points in the tree, just run linear search.
            return self.linear_variant().par_compressive_search(tree, query);
        }

        let mut candidate_radii = SizedHeap::new(None);

        let d = tree.distance_to_uncompressed(query, 0);
        let root = tree.get_cluster_unchecked(0);
        let (car, radius) = (root.cardinality, root.radius);
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((1 + car / 2, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest_id = 0;
        while !tree.get_cluster_unchecked(latest_id).is_leaf() {
            // We have not yet reached a leaf cluster, so we need to explore the children of the current cluster.
            if let Some(child_center_ids) = tree.par_decompress_child_centers(latest_id) {
                let distances = child_center_ids
                    .into_par_iter()
                    .map(|cid| tree.items[cid].1.distance_to_uncompressed(query.borrow(), &tree.metric).map(|d| (cid, d)))
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap_or_else(|_| unreachable!("We just decompressed the child centers."));

                if let Some((cid, d)) = distances.into_iter().min_by_key(|&(_, d)| crate::utils::MinItem((), d)) {
                    let c = tree.get_cluster_unchecked(cid);
                    let (car, radius) = (c.cardinality, c.radius);
                    candidate_radii.push((1, Reverse(d_min(radius, d))));
                    candidate_radii.push((1 + car / 2, Reverse(d)));
                    candidate_radii.push((car, Reverse(d_max(radius, d))));

                    latest_id = cid;
                }
            }
        }

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in arrange_candidate_radii(candidate_radii) {
            if e < self.k {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess::new(d).par_compressive_search(tree, query);
            if hits.len() >= self.k {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.k);
                break;
            }
        }
        hits
    }
}

/// Arranges the candidate radii for RRNN search in a way that ensures non-decreasing expected counts and radii.
fn arrange_candidate_radii<T: DistanceValue>(candidate_radii: SizedHeap<usize, Reverse<T>>) -> Vec<(usize, T)> {
    // Remove all zero-radius candidates
    let mut candidate_radii = candidate_radii
        .take_items()
        .filter_map(|(e, Reverse(d))| if d.is_zero() { None } else { Some((e, d)) })
        .collect::<Vec<_>>();
    // Sort by radius in non-ascending order to ensure can pop the smallest radius as the initial value in the `scan` operation.
    candidate_radii.sort_by_key(|&(_, d)| crate::utils::MinItem((), Reverse(d)));
    let (min_e, min_d) = candidate_radii
        .pop()
        .unwrap_or_else(|| unreachable!("There will always be at least one non-zero candidate radius."));
    // Reverse the order to have candidates in non-decreasing order of radius, and then accumulate the expected counts to ensure they are non-decreasing.
    candidate_radii.reverse();
    // Accumulate the expected counts to ensure they are non-decreasing.
    candidate_radii
        .into_iter()
        .scan((min_e, min_d), |(acc_e, cur_d), (e, d)| {
            // Accumulate the expected counts to ensure they are non-decreasing.
            let result = Some((*acc_e, *cur_d));
            *acc_e += e;
            *cur_d = d;
            result
        })
        .collect()
}
