//! k-NN search by repeated RNN searches with increasing radius.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use rayon::prelude::*;

use crate::{Cluster, Dataset, DistanceValue, ParCluster, ParDataset, SizedHeap};

use super::{
    rnn_clustered::{par_tree_search, tree_search},
    ParSearchAlgorithm, SearchAlgorithm,
};

/// k-NN search by repeated RNN searches with increasing radius.
pub struct KnnRepeatedRnn<T: DistanceValue>(pub usize, pub T);

impl<I, T: DistanceValue, C: Cluster<T>, M: Fn(&I, &I) -> T, D: Dataset<I>> SearchAlgorithm<I, T, C, M, D>
    for KnnRepeatedRnn<T>
{
    fn search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut radius = root.radius_for_k(self.0);

        let [mut confirmed, mut straddlers] = tree_search(
            data,
            metric,
            root,
            query,
            T::from_f32(radius)
                .unwrap_or_else(|| unreachable!("f32 to {} conversion failed", std::any::type_name::<T>())),
        );
        let mut num_confirmed = count_hits(&confirmed);

        while num_confirmed < self.0 {
            let multiplier = if num_confirmed == 0 {
                2.0
            } else {
                lfd_multiplier(&confirmed, &straddlers, self.0)
                    .min(
                        self.1
                            .to_f32()
                            .unwrap_or_else(|| unreachable!("{} to f32 conversion failed", std::any::type_name::<T>())),
                    )
                    .max(1_f32.next_up())
            };
            radius *= multiplier;
            [confirmed, straddlers] = tree_search(
                data,
                metric,
                root,
                query,
                T::from_f32(radius)
                    .unwrap_or_else(|| unreachable!("f32 to {} conversion failed", std::any::type_name::<T>())),
            );
            num_confirmed = count_hits(&confirmed);
        }

        let mut knn = SizedHeap::new(Some(self.0));
        for (leaf, d) in confirmed.into_iter().chain(straddlers) {
            if leaf.is_singleton() {
                knn.extend(leaf.indices().into_iter().map(|i| (d, i)));
            } else {
                knn.extend(
                    data.query_to_many(query, leaf.indices(), metric)
                        .into_iter()
                        .map(|(i, d)| (d, i)),
                );
            }
        }

        knn.items().map(|(d, i)| (i, d)).collect()
    }
}

impl<
        I: Send + Sync,
        T: DistanceValue + Send + Sync,
        C: ParCluster<T>,
        M: (Fn(&I, &I) -> T) + Send + Sync,
        D: ParDataset<I>,
    > ParSearchAlgorithm<I, T, C, M, D> for KnnRepeatedRnn<T>
{
    fn par_search(&self, data: &D, metric: &M, root: &C, query: &I) -> Vec<(usize, T)> {
        let mut radius = root.radius_for_k(self.0);

        let [mut confirmed, mut straddlers] = par_tree_search(
            data,
            metric,
            root,
            query,
            T::from_f32(radius)
                .unwrap_or_else(|| unreachable!("f32 to {} conversion failed", std::any::type_name::<T>())),
        );
        let mut num_confirmed = count_hits(&confirmed);

        while num_confirmed < self.0 {
            let multiplier = if num_confirmed == 0 {
                2.0
            } else {
                lfd_multiplier(&confirmed, &straddlers, self.0)
                    .min(
                        self.1
                            .to_f32()
                            .unwrap_or_else(|| unreachable!("{} to f32 conversion failed", std::any::type_name::<T>())),
                    )
                    .max(1_f32.next_up())
            };
            radius *= multiplier;
            [confirmed, straddlers] = par_tree_search(
                data,
                metric,
                root,
                query,
                T::from_f32(radius)
                    .unwrap_or_else(|| unreachable!("f32 to {} conversion failed", std::any::type_name::<T>())),
            );
            num_confirmed = count_hits(&confirmed);
        }

        let mut knn = SizedHeap::new(Some(self.0));
        for (leaf, d) in confirmed.into_iter().chain(straddlers) {
            if leaf.is_singleton() {
                knn.extend(leaf.indices().into_iter().map(|i| (d, i)));
            } else {
                knn.par_extend(
                    data.par_query_to_many(query, leaf.indices(), metric)
                        .into_par_iter()
                        .map(|(i, d)| (d, i)),
                );
            }
        }

        knn.items().map(|(d, i)| (i, d)).collect()
    }
}

/// Count the total cardinality of the clusters.
fn count_hits<T: DistanceValue, C: Cluster<T>>(hits: &[(&C, T)]) -> usize {
    hits.iter().map(|(c, _)| c.cardinality()).sum()
}

/// Calculate a multiplier for the radius using the LFDs of the clusters.
fn lfd_multiplier<T: DistanceValue, C: Cluster<T>>(confirmed: &[(&C, T)], straddlers: &[(&C, T)], k: usize) -> f32 {
    let (mu, car) = confirmed
        .iter()
        .chain(straddlers.iter())
        .map(|&(c, _)| (c.lfd().recip(), c.cardinality()))
        .fold((0.0, 0), |(lfd, car), (l, c)| (lfd + l, car + c));
    let mu = mu / (confirmed.len() + straddlers.len()) as f32;
    (k as f32 / car as f32).powf(mu).next_up()
}
