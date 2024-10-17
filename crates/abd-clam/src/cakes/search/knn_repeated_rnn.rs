//! k-NN search using a linear scan of the dataset.

use distances::{number::Multiplication, Number};
use rayon::prelude::*;

use crate::{
    cakes::{ParSearchable, Searchable},
    cluster::ParCluster,
    metric::ParMetric,
    Cluster, Metric, SizedHeap,
};

use super::{
    rnn_clustered::{par_tree_search, tree_search},
    ParSearchAlgorithm, SearchAlgorithm,
};

/// k-NN search using a linear scan of the dataset.
pub struct KnnRepeatedRnn<T: Number>(pub usize, pub T);

impl<I, T: Number, C: Cluster<T>, M: Metric<I, T>, D: Searchable<I, T, C, M>> SearchAlgorithm<I, T, C, M, D>
    for KnnRepeatedRnn<T>
{
    fn name(&self) -> &str {
        "KnnRepeatedRnn"
    }

    fn radius(&self) -> Option<T> {
        None
    }

    fn k(&self) -> Option<usize> {
        Some(self.0)
    }

    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let radii_under_k = root.radii_for_k::<f32>(self.0);
        let mut radius: f32 = crate::utils::mean(&radii_under_k);

        let [mut confirmed, mut straddlers] = tree_search(data, metric, root, query, T::from(radius));
        let mut num_confirmed = count_hits(&confirmed);

        while num_confirmed < self.0 {
            let multiplier = if num_confirmed == 0 {
                T::ONE.double().as_f32()
            } else {
                lfd_multiplier(&confirmed, &straddlers, self.0)
                    .min(self.1.as_f32())
                    .max(f32::ONE + f32::EPSILON)
            };
            radius = radius.mul_add(multiplier, T::EPSILON.as_f32());
            [confirmed, straddlers] = tree_search(data, metric, root, query, T::from(radius));
            num_confirmed = count_hits(&confirmed);
        }

        let mut knn = SizedHeap::new(Some(self.0));
        for (leaf, d) in confirmed.into_iter().chain(straddlers) {
            if knn.len() < self.0 // We don't have enough items yet. OR
                // The current farthest hit is farther than the closest
                // potential item in the leaf.
                || d + leaf.radius() <= knn.peek().map_or(T::MAX, |&(d, _)| d)
            {
                if leaf.is_singleton() {
                    knn.extend(leaf.indices().into_iter().map(|i| (d, i)));
                } else {
                    knn.extend(data.query_to_all(metric, query, leaf).map(|(i, d)| (d, i)));
                }
            }
        }
        knn.items().map(|(d, i)| (i, d)).collect()
    }
}

impl<I: Send + Sync, T: Number, C: ParCluster<T>, M: ParMetric<I, T>, D: ParSearchable<I, T, C, M>>
    ParSearchAlgorithm<I, T, C, M, D> for KnnRepeatedRnn<T>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let radii_under_k = root.radii_for_k::<f32>(self.0);
        let mut radius: f32 = crate::utils::mean(&radii_under_k);

        let [mut confirmed, mut straddlers] = par_tree_search(data, metric, root, query, T::from(radius));
        let mut num_confirmed = count_hits(&confirmed);

        while num_confirmed < self.0 {
            let multiplier = if num_confirmed == 0 {
                T::ONE.double().as_f32()
            } else {
                lfd_multiplier(&confirmed, &straddlers, self.0)
                    .min(self.1.as_f32())
                    .max(f32::ONE + f32::EPSILON)
            };
            radius = radius.mul_add(multiplier, T::EPSILON.as_f32());
            [confirmed, straddlers] = par_tree_search(data, metric, root, query, T::from(radius));
            num_confirmed = count_hits(&confirmed);
        }

        let mut knn = SizedHeap::new(Some(self.0));
        for (leaf, d) in confirmed.into_iter().chain(straddlers) {
            if knn.len() < self.0 // We don't have enough items yet. OR
                // The current farthest hit is farther than the closest
                // potential item in the leaf.
                || d + leaf.radius() <= knn.peek().map_or(T::MAX, |&(d, _)| d)
            {
                if leaf.is_singleton() {
                    knn.extend(leaf.indices().into_iter().map(|i| (d, i)));
                } else {
                    knn.par_extend(data.par_query_to_all(metric, query, leaf).map(|(i, d)| (d, i)));
                }
            }
        }
        knn.items().map(|(d, i)| (i, d)).collect()
    }
}

/// Count the total cardinality of the clusters.
fn count_hits<T: Number, C: Cluster<T>>(hits: &[(&C, T)]) -> usize {
    hits.iter().map(|(c, _)| c.cardinality()).sum()
}

/// Calculate a multiplier for the radius using the LFDs of the clusters.
fn lfd_multiplier<T: Number, C: Cluster<T>>(confirmed: &[(&C, T)], straddlers: &[(&C, T)], k: usize) -> f32 {
    let (mu, car) = confirmed
        .iter()
        .chain(straddlers.iter())
        .map(|&(c, _)| (c.lfd().recip(), c.cardinality()))
        .fold((0.0, 0), |(lfd, car), (l, c)| (lfd + l, car + c));
    let mu = mu / (confirmed.len() + straddlers.len()).as_f32();
    (k.as_f32() / car.as_f32()).powf(mu)
}
