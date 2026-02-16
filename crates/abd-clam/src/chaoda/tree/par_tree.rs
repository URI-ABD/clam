//! Parallel tree annotation with anomaly features.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{DistanceValue, Tree, utils::SizedHeap};

use super::super::{AnomalyFeatures, MetaMlModel, normalize_features};

impl<D, M, T, A> Tree<D, M, T, A>
where
    D: Send + Sync,
    M: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`Self::annotate_anomaly_features`].
    pub fn par_annotate_anomaly_features(self) -> Tree<D, M, T, (A, AnomalyFeatures)>
    where
        T: DistanceValue,
    {
        let mut features_map: HashMap<usize, AnomalyFeatures> = core::iter::once((0, AnomalyFeatures::for_root())).collect();
        let mut frontier = vec![0];

        while !frontier.is_empty() {
            let new_features = frontier
                .into_par_iter()
                .filter_map(|id| {
                    if let Ok(parent) = self.get_cluster(id)
                        && let Some(child_center_indices) = parent.child_center_indices()
                    {
                        let parent_features = features_map
                            .get(&id)
                            .unwrap_or_else(|| unreachable!("Missing features for parent with id {id}"));
                        let child_features = child_center_indices.par_iter().filter_map(|&cid| {
                            self.get_cluster(cid)
                                .map(|child| {
                                    let features = parent_features.for_child(parent, child);
                                    (cid, features)
                                })
                                .ok()
                        });
                        Some(child_features)
                    } else {
                        None
                    }
                })
                .flatten()
                .collect::<Vec<_>>();

            frontier = new_features.iter().map(|(id, _)| *id).collect();
            features_map.extend(new_features);
        }

        // Normalize features.
        normalize_features(&mut features_map);

        // Annotate clusters with their features.
        let (items, metric, cluster_map) = self.into_parts();
        let cluster_map = cluster_map
            .into_iter()
            .map(|(id, cluster)| {
                features_map
                    .remove(&id)
                    .map(|features| (id, cluster.compound_annotation(features)))
                    .ok_or_else(|| format!("Missing features for cluster with id {id}"))
            })
            .collect::<Result<_, _>>()
            .unwrap_or_else(|err| unreachable!("All clusters should be successfully annotated. Got error: {err}"));

        if !features_map.is_empty() {
            unreachable!(
                "All clusters should be successfully annotated. Got extra features for cluster ids: {:?}",
                features_map.keys()
            );
        }

        // Construct a new tree with the same structure and items, but with annotated clusters.
        Tree::from_parts(items, metric, cluster_map)
    }
}

impl<D, M, T, A> Tree<D, M, T, (A, AnomalyFeatures)>
where
    D: Send + Sync,
    M: Send + Sync,
    T: Send + Sync,
    A: Send + Sync,
{
    /// Parallel version of [`Self::select_chaoda_clusters`].
    pub fn par_select_chaoda_clusters(&self, predictor: &MetaMlModel, min_depth: usize) -> (Vec<usize>, Vec<usize>) {
        // Rank clusters by their score according to the `predictor`, filtering out clusters that are too shallow.
        let mut rankings = self
            .cluster_map
            .par_iter()
            .filter_map(|(_, c)| if c.depth() >= min_depth { Some((c, predictor.predict(c))) } else { None })
            .collect::<SizedHeap<_, _>>();

        // Greedily select clusters in order of their rank, ignoring ancestors and descendants of previously selected clusters, until there are no more clusters
        // to select.
        let mut selected_clusters = Vec::new();
        let mut covered_items = vec![false; self.root().cardinality];

        while let Some((cluster, _)) = rankings.pop()
            && !covered_items[cluster.items_indices()].iter().any(|&b| b)
        // None of this cluster's items have been covered by previously selected clusters.
        {
            selected_clusters.push(cluster.center_index);
            // Mark all items in this cluster as covered.
            for i in cluster.items_indices() {
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
