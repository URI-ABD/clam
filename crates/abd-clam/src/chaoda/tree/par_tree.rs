//! Parallel tree annotation with anomaly features.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{DistanceValue, Tree, tree::ClusterLocation, utils::SizedHeap};

use super::super::{AnomalyFeatures, meta_ml::MetaMlModel, normalize_features};

impl<Id, I, T, A, M> Tree<Id, I, T, A, M>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::annotate_anomaly_features`].
    pub fn par_annotate_anomaly_features(self) -> Tree<Id, I, T, (A, AnomalyFeatures), M>
    where
        T: DistanceValue,
    {
        let mut features_map: HashMap<usize, AnomalyFeatures> = core::iter::once((0, AnomalyFeatures::for_root())).collect();
        let mut frontier = vec![0];

        while !frontier.is_empty() {
            let new_features = frontier
                .into_par_iter()
                .filter_map(|id| {
                    let cluster = self.get_cluster_unchecked(id);
                    cluster.child_center_indices().map(|child_center_indices| {
                        let parent_features = features_map
                            .get(&id)
                            .unwrap_or_else(|| unreachable!("Missing features for parent with id {id}"));

                        child_center_indices.par_iter().map(|&cid| {
                            let child = self.get_cluster_unchecked(cid);
                            let features = parent_features.for_child(cluster, child);
                            (cid, features)
                        })
                    })
                })
                .flatten()
                .collect::<Vec<_>>();

            frontier = new_features.iter().map(|(id, _)| *id).collect();
            features_map.extend(new_features);
        }

        // Normalize features.
        normalize_features(&mut features_map);

        // Annotate clusters with their features.
        let Self { items, metric } = self;
        let items = items
            .into_iter()
            .map(|(id, item, loc)| {
                let loc = match loc {
                    ClusterLocation::Cluster(c) => {
                        let features = features_map
                            .remove(&c.center_index)
                            .unwrap_or_else(|| unreachable!("Missing features for cluster with id {}", c.center_index));
                        ClusterLocation::Cluster(c.compound_annotation(features))
                    }
                    ClusterLocation::CenterIndex(i) => ClusterLocation::CenterIndex(i),
                };
                (id, item, loc)
            })
            .collect();

        if !features_map.is_empty() {
            unreachable!(
                "All clusters should be successfully annotated. Got extra features for cluster ids: {:?}",
                features_map.keys()
            );
        }

        // Construct a new tree with the same structure and items, but with annotated clusters.
        Tree { items, metric }
    }
}

impl<Id, I, T, A, M> Tree<Id, I, T, (A, AnomalyFeatures), M>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
    M: Send + Sync,
{
    /// Parallel version of [`Self::select_chaoda_clusters`].
    pub(crate) fn par_select_chaoda_clusters(&self, predictor: &MetaMlModel, min_depth: usize) -> (Vec<usize>, Vec<usize>) {
        // Rank clusters by their score according to the `predictor`, filtering out clusters that are too shallow.
        let mut rankings = self
            .items
            .par_iter()
            .filter_map(|(_, _, loc)| loc.as_cluster())
            .filter(|c| c.depth >= min_depth)
            .map(|c| (c, predictor.predict(c)))
            .collect::<SizedHeap<_, _>>();

        // Greedily select clusters in order of their rank, ignoring ancestors and descendants of previously selected clusters, until there are no more clusters
        // to select.
        let mut selected_clusters = Vec::new();
        let mut covered_items = vec![false; self.root().cardinality];

        while let Some((cluster, _)) = rankings.pop()
            && !covered_items[cluster.items_range()].iter().any(|&b| b)
        // None of this cluster's items have been covered by previously selected clusters.
        {
            selected_clusters.push(cluster.center_index);
            // Mark all items in this cluster as covered.
            for i in cluster.items_range() {
                covered_items[i] = true;
            }
        }

        // Collect the indices of items that are not covered by any selected cluster. These are the centers of the ancestors of the selected clusters.
        let uncovered_indices = covered_items
            .iter()
            .enumerate()
            .filter_map(|(i, &covered)| if covered { None } else { Some(i) })
            .collect::<Vec<_>>();

        (selected_clusters, uncovered_indices)
    }
}
