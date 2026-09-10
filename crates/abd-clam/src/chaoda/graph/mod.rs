//! A `Graph` is a collection of `Node`s and the edges between them, and is used for anomaly detection in CHAODA.

use std::collections::HashMap;

use crate::{DistanceValue, Tree, chaoda::AnomalyFeatures, utils::MinItem};

mod component;
mod node;
mod par_component;
mod par_graph;

pub use component::Component;
pub use node::Node;

/// A `Graph` is a collection of `Component`s, each of which is a collection of `Node`s and the edges between them, and is used for anomaly detection in CHAODA.
#[must_use]
pub struct Graph<T: DistanceValue> {
    /// The components in the graph.
    pub components: Vec<Component<T>>,
}

impl<T: DistanceValue> Graph<T> {
    /// Creates a new `Graph` from a `Tree`.
    pub fn from_tree<Id, I, M, A>(tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>, directly_selected: &[usize], ancestors: &[usize]) -> Self
    where
        M: Fn(&I, &I) -> T,
    {
        let mut nodes = directly_selected
            .iter()
            .filter_map(|&index| tree.items[index].2.as_cluster().map(Node::from_cluster).map(|n| (n.direct_center_index(), n)))
            .collect::<HashMap<_, _>>();

        // Place ancestors among nodes.
        for &i in ancestors {
            // Traverse down the tree from the ancestor along the path of nearest child nodes until we reach a node that is already in the graph.
            let mut best_node_index = i; // Start at the ancestor.
            let mut min_distance = T::zero(); // The distance from the ancestor to the nearest node in the graph.
            while let Some(child_ids) = tree.items[best_node_index].2.as_cluster().and_then(|c| c.child_center_indices())
                // Consider all children of the current node and find the nearest one to the ancestor.
                && let Some((nearest_child_index, nearest_child_distance)) = child_ids
                    .iter()
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
            .iter()
            .enumerate()
            .flat_map(|(i, left_index)| {
                directly_selected
                    .iter()
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

        // Create components from nodes and return the graph.
        Self {
            components: Component::from_nodes(nodes),
        }
    }

    /// Returns the number of components in this graph.
    #[must_use]
    pub const fn num_components(&self) -> usize {
        self.components.len()
    }

    /// Returns the number of nodes in this graph.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.iter_components().map(Component::num_nodes).sum()
    }

    /// Returns an iterator over the components in this graph.
    pub fn iter_components(&self) -> impl Iterator<Item = &Component<T>> {
        self.components.iter()
    }

    /// Returns an iterator over all nodes in this graph.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node<T>> {
        self.components.iter().flat_map(Component::iter_nodes)
    }

    /// Returns a reference to the node with the given index, if it exists in this graph.
    #[must_use]
    pub fn get_node(&self, index: usize) -> Option<&Node<T>> {
        self.iter_components().find_map(|c| c.get_node(index))
    }

    /// Returns a mutable reference to a node with the given index, if it exists in this graph.
    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut Node<T>> {
        self.components.iter_mut().find_map(|c| c.get_node_mut(index))
    }

    /// Returns the mean of the `AnomalyFeatures` of all nodes in this graph.
    ///
    /// # Errors
    ///
    /// - If any of the `Cluster`s corresponding to the nodes in this graph was not found in the `Tree`.
    #[expect(clippy::cast_precision_loss)]
    pub(crate) fn mean_anomaly_features<Id, I, A, M>(&self, tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>) -> AnomalyFeatures {
        let mut acc_features = self.iter_nodes().fold(AnomalyFeatures::default(), |mut acc, node| {
            let ann = tree.get_cluster_unchecked(node.direct_center_index()).annotation.1;
            acc.cardinality_ratio += ann.cardinality_ratio;
            acc.radius_ratio += ann.radius_ratio;
            acc.lfd_ratio += ann.lfd_ratio;
            acc.ema_cardinality_ratio += ann.ema_cardinality_ratio;
            acc.ema_radius_ratio += ann.ema_radius_ratio;
            acc.ema_lfd_ratio += ann.ema_lfd_ratio;
            acc
        });

        let n = self.num_nodes() as f64;
        acc_features.cardinality_ratio /= n;
        acc_features.radius_ratio /= n;
        acc_features.lfd_ratio /= n;
        acc_features.ema_cardinality_ratio /= n;
        acc_features.ema_radius_ratio /= n;
        acc_features.ema_lfd_ratio /= n;
        acc_features
    }
}

/// Computes the distance between two center items in the tree and, if the distance is less than or equal to the sum of the radii of the two clusters, returns
/// the distance. Otherwise, returns `None`.
pub fn compute_edge_distance<Id, I, T, A, M>(tree: &Tree<Id, I, T, (A, AnomalyFeatures), M>, left: &Node<T>, right: &Node<T>) -> Option<T>
where
    M: Fn(&I, &I) -> T,
    T: DistanceValue,
{
    let d = (tree.metric)(&tree.items[left.direct_center_index()].1, &tree.items[right.direct_center_index()].1);
    let t = left.radius + right.radius;
    if d <= t { Some(d) } else { None }
}
