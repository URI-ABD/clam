//! A `Graph` is a collection of `Node`s and the edges between them, and is used for anomaly detection in CHAODA.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::{DistanceValue, Tree, chaoda::AnomalyFeatures, utils::MinItem};

use super::{Component, Graph, Node, compute_edge_distance};

impl<T: DistanceValue + Send + Sync> Graph<T> {
    /// Parallel version of [`Graph::from_tree`].
    pub fn par_from_tree<Id, I, M, A>(tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>, directly_selected: &[usize], ancestors: &[usize]) -> Self
    where
        Id: Send + Sync,
        I: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        A: Send + Sync,
    {
        let mut nodes = directly_selected
            .par_iter()
            .filter_map(|&index| tree.items[index].2.as_cluster().map(Node::from_cluster).map(|n| (n.direct_center_index(), n)))
            .collect::<HashMap<_, _>>();

        // Place ancestors among nodes.
        let placements = ancestors
            .par_iter()
            .map(|&i| {
                // Traverse down the tree from the ancestor along the path of nearest child nodes until we reach a node that is already in the graph.
                let mut best_node_index = i; // Start at the ancestor.
                let mut min_distance = T::zero(); // The distance from the ancestor to the nearest node in the graph.
                while let Some(child_ids) = tree.items[best_node_index].2.as_cluster().and_then(|c| c.child_center_indices())
                    // Consider all children of the current node and find the nearest one to the ancestor.
                    && let Some((nearest_child_index, nearest_child_distance)) = child_ids
                        .par_iter()
                        .map(|&j| {
                            // Compute the distance from the ancestor to the center of the child cluster.
                            let d = (tree.metric)(&tree.items[i].1, &tree.items[j].1);
                            (j, d)
                        })
                        // Find the child with the minimum distance to the ancestor.
                        .min_by_key(|&(_, d)| MinItem((), d))
                    // If the nearest child is not already in the graph, continue traversing down the tree.
                    && !nodes.contains_key(&nearest_child_index)
                {
                    best_node_index = nearest_child_index;
                    min_distance = nearest_child_distance;
                }
                (i, best_node_index, min_distance)
            })
            .collect::<Vec<_>>();

        for (i, best_node_index, min_distance) in placements {
            // Add the ancestor index to the node that we found in the graph.
            if let Some(node) = nodes.get_mut(&best_node_index) {
                node.add_ancestor_center(i);
                if min_distance > node.radius {
                    node.radius = min_distance;
                }
            }
        }

        // Compute edges between nodes.
        let edges = directly_selected
            .par_iter()
            .enumerate()
            .flat_map(|(i, left_index)| {
                directly_selected
                    .par_iter()
                    .skip(i + 1)
                    .copied()
                    .filter_map(|right_index| compute_edge_distance(tree, &nodes[left_index], &nodes[&right_index]).map(|d| (*left_index, right_index, d)))
            })
            .collect::<Vec<_>>();

        // Add edges to nodes.
        for (i, j, d) in edges {
            if let Some(left) = nodes.get_mut(&i) {
                left.add_edge(j, d);
            }
            if let Some(right) = nodes.get_mut(&j) {
                right.add_edge(i, d);
            }
        }

        Self {
            components: Component::from_nodes(nodes),
        }
    }

    /// Returns a parallel iterator over the components in this graph.
    #[must_use]
    pub fn par_iter_components(&self) -> impl ParallelIterator<Item = &Component<T>> {
        self.components.par_iter()
    }

    /// Returns a parallel iterator over all nodes in this graph.
    #[must_use]
    pub fn par_iter_nodes(&self) -> impl ParallelIterator<Item = &Node<T>> {
        self.components.par_iter().flat_map(Component::par_iter_nodes)
    }
}
