//! K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.

use std::cmp::Reverse;

use rayon::prelude::*;

use crate::{DistanceValue, Tree, utils::SizedHeap};

use super::super::{ParSearch, RnnChess, Search, d_max, d_min};

/// K-nearest neighbors (KNN) search using the Repeated Ranged Nearest Neighbor (RRNN) algorithm.
///
/// The field is the number of nearest neighbors to find (k).
pub struct KnnRrnn(pub usize);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for KnnRrnn
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("KnnRrnn(k={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let radius = root.radius();

        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree.items.iter().enumerate().map(|(i, (_, item))| (i, (tree.metric())(query, item))).collect();
        }

        let mut candidate_radii = SizedHeap::<usize, Reverse<T>>::new(None);

        let d = (tree.metric)(query, &tree.items[0].1);
        let car = tree.cardinality();
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((car.half() + 1, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest = root;
        while !latest.is_leaf() {
            if let Some((child, d)) = tree.children_of(latest).and_then(|children| {
                children
                    .into_iter()
                    .map(|child| (child, (tree.metric)(query, &tree.items[child.center_index()].1)))
                    .min_by_key(|&(_, d)| crate::utils::MinItem((), d))
            }) {
                let car = child.cardinality();
                let radius = child.radius();
                candidate_radii.push((1, Reverse(d_min(radius, d))));
                candidate_radii.push((car.half() + 1, Reverse(d)));
                candidate_radii.push((car, Reverse(d_max(radius, d))));

                latest = child;
            }
        }

        // Get the non-zero radii in sorted order.
        let candidate_radii = {
            let mut candidate_radii = candidate_radii
                .take_items()
                .filter_map(|(e, Reverse(d))| if d.is_zero() { None } else { Some((e, d)) })
                .collect::<Vec<_>>();
            candidate_radii.sort_by_key(|&(_, d)| crate::utils::MinItem((), Reverse(d)));
            let (min_e, min_d) = candidate_radii
                .pop()
                .unwrap_or_else(|| unreachable!("There will always be at least one non-zero candidate radius."));
            candidate_radii.reverse();
            candidate_radii
                .into_iter()
                .scan((min_e, min_d), |(acc_e, cur_d), (e, d)| {
                    // Accumulate the expected counts to ensure they are non-decreasing.
                    let result = Some((*acc_e, *cur_d));
                    *acc_e += e;
                    *cur_d = d;
                    result
                })
                .collect::<Vec<_>>()
        };

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in candidate_radii {
            if e < self.0 {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess(d).search(tree, query);
            if hits.len() >= self.0 {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.0);
                break;
            }
        }
        hits
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for KnnRrnn
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let radius = root.radius();

        if self.0 > tree.cardinality() {
            // If k is greater than the number of points in the tree, return all items with their distances.
            return tree
                .items
                .par_iter()
                .enumerate()
                .map(|(i, (_, item))| (i, (tree.metric())(query, item)))
                .collect();
        }

        let mut candidate_radii = SizedHeap::<usize, Reverse<T>>::new(None);

        let d = (tree.metric)(query, &tree.items[root.center_index()].1);
        let car = tree.cardinality();
        candidate_radii.push((1, Reverse(d_min(radius, d))));
        candidate_radii.push((car.half() + 1, Reverse(d)));
        candidate_radii.push((car, Reverse(d_max(radius, d))));

        let mut latest = root;
        while !latest.is_leaf() {
            if let Some((child, d)) = tree.children_of(latest).and_then(|children| {
                children
                    .into_par_iter()
                    .map(|child| (child, (tree.metric)(query, &tree.items[child.center_index()].1)))
                    .min_by_key(|&(_, d)| crate::utils::MinItem((), d))
            }) {
                let car = child.cardinality();
                let radius = child.radius();
                candidate_radii.push((1, Reverse(d_min(radius, d))));
                candidate_radii.push((car.half() + 1, Reverse(d)));
                candidate_radii.push((car, Reverse(d_max(radius, d))));

                latest = child;
            }
        }

        // Get the non-zero radii in sorted order.
        let candidate_radii = {
            let mut candidate_radii = candidate_radii
                .take_items()
                .filter_map(|(e, Reverse(d))| if d.is_zero() { None } else { Some((e, d)) })
                .collect::<Vec<_>>();
            candidate_radii.sort_by_key(|&(_, d)| crate::utils::MinItem((), Reverse(d)));
            let (min_e, min_d) = candidate_radii
                .pop()
                .unwrap_or_else(|| unreachable!("There will always be at least one non-zero candidate radius."));
            candidate_radii.reverse();
            candidate_radii
                .into_iter()
                .scan((min_e, min_d), |(acc_e, cur_d), (e, d)| {
                    // Accumulate the expected counts to ensure they are non-decreasing.
                    let result = Some((*acc_e, *cur_d));
                    *acc_e += e;
                    *cur_d = d;
                    result
                })
                .collect::<Vec<_>>()
        };

        // Search for neighbors within the candidate radii until we find at least k neighbors.
        let mut hits = Vec::new();
        for (e, d) in candidate_radii {
            if e < self.0 {
                // If the candidate radius is too small to expect k neighbors, skip it.
                continue;
            }

            hits = RnnChess(d).par_search(tree, query);
            if hits.len() >= self.0 {
                hits.sort_by_key(|&(_, d)| crate::utils::MinItem((), d));
                hits.truncate(self.0);
                break;
            }
        }
        hits
    }
}
