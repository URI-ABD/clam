//! Ranged Nearest Neighbors (RNN) search using the CHESS algorithm.

use rayon::prelude::*;

use crate::{
    DistanceValue, Tree,
    cakes::{ParSearch, Search},
};

/// Ranged Nearest Neighbors search using the CHESS algorithm.
///
/// The field is the radius of the query ball to search within.
pub struct RnnChess<T: DistanceValue>(pub T);

impl<Id, I, T, A, M> Search<Id, I, T, A, M> for RnnChess<T>
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    fn name(&self) -> String {
        format!("RnnChess(radius={})", self.0)
    }

    fn search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = tree.distance_to(query, 0); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while !frontier.is_empty() {
            let (new_frontier, new_hits) = frontier
                .into_iter()
                .map(|(d, cluster)| {
                    if d <= self.0 + cluster.radius() {
                        // Overlapping volume

                        let mut local_hits = Vec::new();

                        if d <= self.0 {
                            // Center is within query radius
                            local_hits.push((cluster.center_index(), d));
                        }

                        if self.0 >= d + cluster.radius() {
                            // Fully subsumed cluster
                            tree.distances_to_items_in_subtree(query, cluster).for_each(|item| local_hits.push(item));
                            (Vec::new(), local_hits)
                        } else if let Some(children) = tree.children_of(cluster) {
                            let mut new_frontier = Vec::new();
                            for child in children {
                                let d_child = tree.distance_to_center(query, child);
                                // Check children for overlap
                                if d_child <= self.0 + child.radius() {
                                    new_frontier.push((d_child, child));
                                }
                            }
                            (new_frontier, local_hits)
                        } else {
                            // Leaf cluster and not fully subsumed
                            tree.distances_to_items_in_subtree(query, cluster)
                                .filter(|(_, dist)| *dist <= self.0)
                                .for_each(|item| local_hits.push(item));
                            (Vec::new(), local_hits)
                        }
                    } else {
                        (Vec::new(), Vec::new())
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            frontier = new_frontier.into_iter().flatten().collect();
            hits.extend(new_hits.into_iter().flatten());
        }

        hits
    }
}

impl<Id, I, T, A, M> ParSearch<Id, I, T, A, M> for RnnChess<T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    fn par_search(&self, tree: &Tree<Id, I, T, A, M>, query: &I) -> Vec<(usize, T)> {
        let root = tree.root();
        let d_root = tree.distance_to(query, 0); // root center index is always 0
        // Check to see if there is any overlap with the root
        if d_root > self.0 + root.radius() {
            return Vec::new(); // No overlap
        }

        let mut hits = Vec::new();

        let mut frontier = vec![(d_root, root)];
        while !frontier.is_empty() {
            let (new_frontier, new_hits) = frontier
                .into_par_iter()
                .map(|(d, cluster)| {
                    if d <= self.0 + cluster.radius() {
                        // Overlapping volume

                        let mut local_hits = Vec::new();

                        if d <= self.0 {
                            // Center is within query radius
                            local_hits.push((cluster.center_index(), d));
                        }

                        if self.0 >= d + cluster.radius() {
                            // Fully subsumed cluster
                            for item in tree.par_distances_to_items_in_subtree(query, cluster).collect::<Vec<_>>() {
                                local_hits.push(item);
                            }
                            (Vec::new(), local_hits)
                        } else if let Some(children) = tree.children_of(cluster) {
                            let new_frontier = children
                                .into_par_iter()
                                .filter_map(|child| {
                                    let d_child = tree.distance_to_center(query, child);
                                    // Check children for overlap
                                    if d_child <= self.0 + child.radius() { Some((d_child, child)) } else { None }
                                })
                                .collect::<Vec<_>>();

                            (new_frontier, local_hits)
                        } else {
                            // Leaf cluster and not fully subsumed
                            for item in tree
                                .par_distances_to_items_in_subtree(query, cluster)
                                .filter(|(_, dist)| *dist <= self.0)
                                .collect::<Vec<_>>()
                            {
                                local_hits.push(item);
                            }

                            (Vec::new(), local_hits)
                        }
                    } else {
                        (Vec::new(), Vec::new())
                    }
                })
                .unzip::<_, _, Vec<_>, Vec<_>>();

            frontier = new_frontier.into_iter().flatten().collect();
            hits.extend(new_hits.into_iter().flatten());
        }

        hits
    }
}
