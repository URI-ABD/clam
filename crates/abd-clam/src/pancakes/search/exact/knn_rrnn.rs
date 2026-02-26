//! K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.

use std::cmp::Reverse;

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{KnnRrnn, d_max, d_min},
    pancakes::{Codec, MaybeCompressed},
    utils::SizedHeap,
};

use super::super::{CompressiveSearch, ParCompressiveSearch, RnnChess};

impl<Id, I, T, A, M> CompressiveSearch<Id, I, T, A, M> for KnnRrnn
where
    I: Codec,
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.decompress_subtree(0);
            return tree
                .items
                .iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }

        let mut candidate_radii = SizedHeap::new(None);

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        let (car, radius) = tree.get_cluster(0).map(|c| (c.cardinality(), c.radius))?;
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((car.half() + 1, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest_id = 0;
        while tree.get_cluster(latest_id).map(|c| !c.is_leaf())? {
            // We have not yet reached a leaf cluster, so we need to explore the children of the current cluster.
            if let Ok(child_center_ids) = tree.decompress_child_centers(latest_id) {
                let distances = child_center_ids
                    .into_iter()
                    .map(|cid| tree.items[cid].1.distance_to_query(query, &tree.metric).map(|d| (cid, d)))
                    .collect::<Result<Vec<_>, _>>()?;

                if let Some((cid, d)) = distances.into_iter().min_by_key(|&(_, d)| crate::utils::MinItem((), d)) {
                    let (car, radius) = tree.get_cluster(cid).map(|c| (c.cardinality(), c.radius))?;
                    candidate_radii.push((1, Reverse(d_min(radius, d))));
                    candidate_radii.push((car.half() + 1, Reverse(d)));
                    candidate_radii.push((car, Reverse(d_max(radius, d))));

                    latest_id = cid;
                }
            }
        }

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in arrange_candidate_radii(candidate_radii) {
            if e < self.0 {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess(d).compressive_search(tree, query)?;
            if hits.len() >= self.0 {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.0);
                break;
            }
        }
        Ok(hits)
    }
}

impl<Id, I, T, A, M> ParCompressiveSearch<Id, I, T, A, M> for KnnRrnn
where
    Id: Send + Sync,
    I: Codec + Send + Sync,
    I::Compressed: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_compressive_search(&self, tree: &mut Tree<Id, MaybeCompressed<I>, T, A, M>, query: &I) -> Result<Vec<(usize, T)>, String> {
        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            tree.par_decompress_subtree(0);
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| item.distance_to_query(query, &tree.metric).map(|d| (i, d)))
                .collect();
        }

        let mut candidate_radii = SizedHeap::new(None);

        let d = tree.items[0].1.distance_to_query(query, &tree.metric)?;
        let (car, radius) = tree.get_cluster(0).map(|c| (c.cardinality(), c.radius))?;
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((car.half() + 1, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest_id = 0;
        while tree.get_cluster(latest_id).map(|c| !c.is_leaf())? {
            // We have not yet reached a leaf cluster, so we need to explore the children of the current cluster.
            if let Some(child_center_ids) = tree.par_decompress_child_centers(latest_id)? {
                let distances = child_center_ids
                    .into_par_iter()
                    .map(|cid| tree.items[cid].1.distance_to_query(query, &tree.metric).map(|d| (cid, d)))
                    .collect::<Result<Vec<_>, _>>()?;

                if let Some((cid, d)) = distances.into_iter().min_by_key(|&(_, d)| crate::utils::MinItem((), d)) {
                    let (car, radius) = tree.get_cluster(cid).map(|c| (c.cardinality(), c.radius))?;
                    candidate_radii.push((1, Reverse(d_min(radius, d))));
                    candidate_radii.push((car.half() + 1, Reverse(d)));
                    candidate_radii.push((car, Reverse(d_max(radius, d))));

                    latest_id = cid;
                }
            }
        }

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in arrange_candidate_radii(candidate_radii) {
            if e < self.0 {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess(d).par_compressive_search(tree, query)?;
            if hits.len() >= self.0 {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.0);
                break;
            }
        }
        Ok(hits)
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
