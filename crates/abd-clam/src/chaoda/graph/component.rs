//! A single connected component in a CHAODA graph.

use std::collections::HashMap;

use ndarray::prelude::*;

use crate::DistanceValue;

use super::Node;

/// A single connected component in a CHAODA graph.
#[must_use]
pub struct Component<T: DistanceValue> {
    /// The nodes in the component.
    pub(crate) nodes: HashMap<usize, Node<T>>,
}

impl<T: DistanceValue> Component<T> {
    /// Creates new `Component`s from nodes that have their edges assigned.
    #[must_use]
    pub fn from_nodes(mut all_nodes: HashMap<usize, Node<T>>) -> Vec<Self> {
        let mut components = Vec::new();

        while let Some((&start_index, _)) = all_nodes.iter().next() {
            let mut nodes = HashMap::new();
            let mut frontier = vec![start_index];
            while let Some(next_index) = frontier.pop()
                && let Some(node) = all_nodes.remove(&next_index)
            {
                frontier.extend(node.edges.iter().map(|&(neighbor_index, _)| neighbor_index));
                nodes.insert(next_index, node);
            }
            components.push(Self { nodes });
        }

        components
    }

    /// Returns the number of nodes in this component.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns an iterator over the nodes in this component.
    pub fn iter_nodes(&self) -> impl Iterator<Item = &Node<T>> {
        self.nodes.values()
    }

    /// Returns a node in this component by its index, if it exists.
    #[must_use]
    pub fn get_node(&self, index: usize) -> Option<&Node<T>> {
        self.nodes.get(&index)
    }

    /// Returns a mutable reference to a node in this component by its index, if it exists.
    pub fn get_node_mut(&mut self, index: usize) -> Option<&mut Node<T>> {
        self.nodes.get_mut(&index)
    }

    /// Returns the transition probability matrix for this component.
    ///
    /// The matrix is represented as an Array2, where the entry at (i, j) is the probability of transitioning from node i to node j. This probability is
    /// inversely proportional to the distance between the nodes, normalized so that the probabilities for each node sum to 1.
    ///
    /// # Returns
    ///
    /// A tuple of:
    ///
    /// - An Array2 representing the transition probability matrix.
    /// - A Vec of references to the nodes, mapping the row/column indices of the matrix to the corresponding node in the component.
    #[must_use]
    pub fn transition_probability_matrix(&self) -> (Array2<f64>, Vec<&Node<T>>) {
        let nodes_vec = self.nodes.iter().map(|(&index, node)| (index, node)).collect::<Vec<_>>();
        let mut matrix = Array2::<f64>::zeros((self.num_nodes(), self.num_nodes()));

        for (i, &(_, left)) in nodes_vec.iter().enumerate() {
            let total_weight = left.edges.iter().filter_map(|&(_, distance)| distance.to_f64()).sum::<f64>().recip();
            for &(right_index, weight) in left.iter_edges() {
                if let Some(j) = nodes_vec.iter().position(|&(index, _)| index == right_index)
                    && let Some(weight) = weight.to_f64()
                {
                    matrix[[i, j]] = weight * total_weight;
                }
            }
        }

        let nodes = nodes_vec.into_iter().map(|(_, node)| node).collect::<Vec<_>>();
        (matrix, nodes)
    }

    /// Returns a nested vector of the other nodes reachable from the given node, where the inner vector at index `i` contains the nodes reachable from the
    /// given node in exactly `i` steps along the shortest path.
    pub fn reachable_nodes_by_steps(&self, node: &Node<T>) -> Vec<Vec<&Node<T>>> {
        let mut reachable = Vec::new();

        let mut unvisited = self.nodes.iter().map(|(&index, node)| (index, node)).collect::<HashMap<_, _>>();
        let mut frontier = node.iter_edges().filter_map(|(i, _)| unvisited.get(i)).copied().collect::<Vec<_>>();
        // If there are still unvisited nodes and the frontier is not empty, we can continue to explore the component.
        while !(unvisited.is_empty() || frontier.is_empty()) {
            // The current frontier represents the nodes reachable in the current number of steps by the shortest path.
            reachable.push(frontier.clone());

            // Remove the nodes in the frontier from the unvisited set.
            for &node in &frontier {
                unvisited.remove(&node.direct_center_index());
            }
            // Construct the next frontier by collecting the neighbors of the nodes in the current frontier that are still unvisited.
            frontier = frontier
                .into_iter()
                .flat_map(|node| node.iter_edges().filter_map(|(i, _)| unvisited.get(i)).copied())
                .collect::<Vec<_>>();
        }

        reachable
    }
}
