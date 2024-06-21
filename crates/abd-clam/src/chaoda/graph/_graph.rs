//! The `Graph` module contains the `Graph` struct and related types, which are used to represent
//! a collection of clusters and edges, i.e. connections between overlapping clusters.
//!
//! The `Graph`s are used in anomaly detection and visualization.

use core::hash::{Hash, Hasher};

use std::collections::{HashMap, HashSet};

use distances::Number;

use crate::{Cluster, Dataset, Instance, Tree};

use super::{
    criteria::{detect_edges, select_clusters},
    MetaMLScorer, Vertex,
};

/// A set of clusters with references to clusters in a graph.
pub type VertexSet<'a, U> = HashSet<&'a Vertex<U>>;

/// A set of edges with references to edges in a graph.
pub type EdgeSet<'a, U> = HashSet<Edge<'a, U>>;

/// A map that represents the adjacency relationship between clusters.
pub type AdjacencyMap<'a, U> = HashMap<&'a Vertex<U>, VertexSet<'a, U>>;

/// A map that associates clusters with lists of frontier sizes.
pub type FrontierSizes<'a, U> = HashMap<&'a Vertex<U>, Vec<usize>>;

/// Two `Cluster`s have an `Edge` between them if they have overlapping volumes.
///
/// In CLAM, all `Edge`s are bi-directional.
#[derive(Debug, Clone)]
pub struct Edge<'a, U: Number> {
    /// A reference to the first `Cluster` connected by this `Edge`.
    left: &'a Vertex<U>,
    /// A reference to the second `Cluster` connected by this `Edge`.
    right: &'a Vertex<U>,

    /// The distance between the two `Cluster`s connected by this `Edge`.
    distance: U,
}

impl<'a, U: Number> PartialEq for Edge<'a, U> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

/// Two `Edge`s are equal if they connect the same two `Cluster`s.
impl<'a, U: Number> Eq for Edge<'a, U> {}

impl<'a, U: Number> std::fmt::Display for Edge<'a, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:} -- {:}", self.left, self.right)
    }
}

impl<'a, U: Number> Hash for Edge<'a, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{self}").hash(state);
    }
}

impl<'a, U: Number> Edge<'a, U> {
    /// Creates a new `Edge` between the given `Cluster`s with the specified distance.
    ///
    /// The user should ensure that the provided `Cluster`s are appropriately related to have an edge
    /// between them. The `Edge` is always created as a bi-directional edge between the two `Cluster`s.
    ///
    /// # Arguments
    ///
    /// * `left`: The first `Cluster` connected by the `Edge`.
    /// * `right`: The second `Cluster` connected by the `Edge`.
    /// * `distance`: The distance between the two `Cluster`s.
    ///
    /// # Returns
    ///
    /// A new `Edge` connecting the provided `Cluster`s with the given distance.
    pub fn new(left: &'a Vertex<U>, right: &'a Vertex<U>, distance: U) -> Self {
        if left < right {
            Self { left, right, distance }
        } else {
            Self {
                left: right,
                right: left,
                distance,
            }
        }
    }

    /// Checks if this edge contains the given `Cluster` at one of its ends.
    ///
    /// # Arguments
    ///
    /// * `c`: The `Cluster` to check if it's at one of the ends of the edge.
    ///
    /// # Returns
    ///
    /// Returns `true` if the `Cluster` is found at either end of the edge, `false` otherwise.
    pub fn contains(&self, c: &Vertex<U>) -> bool {
        c == self.left || c == self.right
    }

    /// Returns a 2-slice containing the `Cluster`s at the two ends of this `Edge`.
    ///
    /// # Returns
    ///
    /// A 2-slice containing the `Cluster`s at the left and right ends of the edge.
    pub const fn clusters(&self) -> [&Vertex<U>; 2] {
        [self.left, self.right]
    }

    /// Retrieves a reference to the `Cluster` at the `left` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the left end of the `Edge`.
    pub const fn left(&self) -> &'a Vertex<U> {
        self.left
    }

    /// Retrieves a reference to the `Cluster` at the `right` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the right end of the `Edge`.
    pub const fn right(&self) -> &'a Vertex<U> {
        self.right
    }

    /// Gets the distance between the two `Cluster`s connected by this `Edge`.
    ///
    /// # Returns
    ///
    /// The distance value representing the length between the two connected clusters.
    pub const fn distance(&self) -> U {
        self.distance
    }

    /// Checks whether this is an edge from a `Cluster` to itself.
    ///
    /// # Returns
    ///
    /// - `true` if the edge connects a `Cluster` to itself, indicating a circular relationship.
    /// - `false` if the edge connects two distinct clusters.
    pub fn is_circular(&self) -> bool {
        self.left == self.right
    }

    /// Returns the neighbor of the given `Cluster` in this `Edge`.
    ///
    /// # Arguments
    ///
    /// * `c`: The `Cluster` for which to find the neighbor.
    ///
    /// # Returns
    ///
    /// A reference to the neighboring `Cluster` connected by this `Edge`.
    ///
    /// # Errors
    ///
    /// Returns an error if `c` is not one of the `Cluster`s connected by this `Edge`.
    pub fn neighbor(&self, c: &Vertex<U>) -> Result<&Vertex<U>, String> {
        if c == self.left {
            Ok(self.right)
        } else if c == self.right {
            Ok(self.left)
        } else {
            Err(format!("Cluster {c} is not in this edge {self}."))
        }
    }
}

/// A `Graph` represents a collection of `Cluster`s and `Edge`s, i.e.
/// connections between overlapping `Cluster`s.
///
/// TODO: Add more info on what graphs we useful for.
#[derive(Debug, Clone)]
pub struct Graph<'a, U: Number> {
    /// A set of `Cluster`s in the graph.
    clusters: VertexSet<'a, U>,
    /// A set of `Edge`s representing connections between `Cluster`s in the graph.
    edges: EdgeSet<'a, U>,
    /// A map that represents the adjacency relationships between clusters.
    adjacency_map: AdjacencyMap<'a, U>,
    /// The total population represented by the clusters in the graph.
    population: usize,
    /// The minimum depth in the hierarchy of clusters.
    min_depth: usize,
    /// The maximum depth in the hierarchy of clusters.
    max_depth: usize,
    /// An ordered list of references to `Cluster`s in the graph.
    ordered_clusters: Vec<&'a Vertex<U>>,
    /// A distance matrix representing distances between clusters.
    distance_matrix: Option<Vec<Vec<U>>>,
    /// An adjacency matrix representing adjacency relationships between clusters.
    adjacency_matrix: Option<Vec<Vec<bool>>>,
    /// A mapping of clusters to their respective frontier sizes.
    #[allow(dead_code)]
    frontier_sizes: Option<FrontierSizes<'a, U>>, // TODO: Bench when replacing with DashMap
}

impl<'a, U: Number> Graph<'a, U> {
    /// Creates a new `Graph` from the provided `tree`.
    ///
    /// This function takes a tree structure and produces a graph based on selected clusters and edges.
    ///
    /// # Arguments
    ///
    /// * `tree`: The `Tree` instance used as the source data for building the `Graph`.
    /// * `scorer_function`:  A function that assigns a score to a cluster indicating its suitability for inclusion in the graph.
    /// * `min_depth`: The minimum depth of clusters to be considered in the graph. It is recommended to use a default `min_depth` value of 4 if unsure.
    ///
    /// # Returns
    ///
    /// Returns a `Result` with a new `Graph` instance constructed from the selected clusters and edges
    /// if the operation succeeds. Otherwise, returns an `Err` containing an error message.
    ///
    /// # Errors
    ///
    /// This function returns an error under the following conditions:
    ///
    /// - If the selected clusters are empty, indicating that a graph cannot be created with no clusters.
    /// - If an edge refers to a cluster that is not part of the selected clusters.
    ///
    pub fn from_tree<I: Instance, D: Dataset<I, U>>(
        tree: &'a Tree<I, U, D, Vertex<U>>,
        scorer_function: &MetaMLScorer,
        min_depth: usize,
    ) -> Result<Self, String> {
        let selected_clusters = select_clusters(tree.root(), scorer_function, min_depth)?;

        let edges = detect_edges(&selected_clusters, tree.data());
        Graph::from_clusters_and_edges(selected_clusters, edges)
    }

    /// Creates a new `Graph` from the provided set of `clusters` and `edges`.
    ///
    /// # Arguments
    ///
    /// * `clusters`: The set of `Cluster`s used to build the `Graph`.
    /// * `edges`: The set of `Edge`s used to build the `Graph`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` with a new `Graph` instance constructed from the provided clusters and edges
    /// if the operation succeeds. Otherwise, returns an `Err` containing an error message.
    ///
    /// # Errors
    ///
    /// This function returns an error under the following conditions:
    ///
    /// - If the provided `clusters` set is empty, indicating that a graph cannot be created with no clusters.
    /// - If an edge refers to a cluster that is not in the `clusters` set.
    ///
    fn from_clusters_and_edges(clusters: VertexSet<'a, U>, edges: EdgeSet<'a, U>) -> Result<Self, String> {
        if clusters.is_empty() {
            return Err("Cannot create a graph with no clusters.".to_string());
        }

        let (population, min_depth, max_depth) =
            clusters
                .iter()
                .fold((0, usize::MAX, 0), |(population, min_depth, max_depth), &c| {
                    (
                        population + c.cardinality(),
                        std::cmp::min(min_depth, c.depth()),
                        std::cmp::max(max_depth, c.depth()),
                    )
                });

        let adjacency_map = {
            let mut adjacency_map: AdjacencyMap<U> = clusters.iter().map(|&c| (c, HashSet::new())).collect();
            for e in &edges {
                adjacency_map
                    .get_mut(e.left())
                    .ok_or_else(|| format!("Left cluster not found: {:?}", e.left()))?
                    .insert(e.right());

                adjacency_map
                    .get_mut(e.right())
                    .ok_or_else(|| format!("Right cluster not found: {:?}", e.right()))?
                    .insert(e.left());
            }
            adjacency_map
        };
        let mut ordered_clusters = clusters.iter().copied().collect::<Vec<_>>();
        ordered_clusters.sort();

        Ok(Self {
            clusters,
            edges,
            adjacency_map,
            population,
            min_depth,
            max_depth,
            ordered_clusters,
            distance_matrix: None,
            adjacency_matrix: None,
            frontier_sizes: None,
        })
    }

    /// Computes the distance matrix for the clusters in the graph.
    ///
    /// The distance matrix is a square matrix where each element represents the
    /// distance between two clusters in the graph. If there is no edge (connection) between
    /// two clusters, the distance is set to zero. The matrix is represented as a
    /// two-dimensional vector, where `matrix[i][j]` holds the distance between cluster `i` and
    /// cluster `j`. The matrix is square with dimensions equal to the number of clusters in the graph.
    ///
    /// # Returns
    ///
    /// A two-dimensional vector representing the distance matrix.
    fn compute_distance_matrix(&self) -> Vec<Vec<U>> {
        let indices: HashMap<_, _> = self.ordered_clusters.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut matrix: Vec<Vec<U>> = vec![vec![U::zero(); self.vertex_cardinality()]; self.vertex_cardinality()];
        self.edges.iter().for_each(|e| {
            let i = *indices
                .get(e.left())
                .unwrap_or_else(|| unreachable!("We asserted all clusters are in the edge set when building the graph"));
            let j = *indices
                .get(e.right())
                .unwrap_or_else(|| unreachable!("We asserted all clusters are in the edge set when building the graph"));
            matrix[i][j] = e.distance();
            matrix[j][i] = e.distance();
        });
        matrix
    }

    /// Computes the distance matrix for the `Graph` and stores it as an
    /// internal property.
    ///
    /// # Returns
    ///
    /// A new `Graph` instance with the updated distance matrix.
    #[must_use]
    pub fn with_distance_matrix(mut self) -> Self {
        self.distance_matrix = Some(self.compute_distance_matrix());
        self
    }

    /// Computes the adjacency matrix for the `Graph` and stores it as an
    /// internal property. If the distance matrix is not already computed, it
    /// calls `with_distance_matrix` to calculate it.
    ///
    /// # Returns
    ///
    /// A new `Graph` instance with the updated adjacency matrix.
    ///
    #[must_use]
    pub fn with_adjacency_matrix(mut self) -> Self {
        if self.distance_matrix.is_none() {
            self = self.with_distance_matrix();
        }

        self.adjacency_matrix = Some(self.distance_matrix().map_or_else(
            || unreachable!("with_distance_matrix is guaranteed to be called"),
            |matrix| {
                matrix
                    .iter()
                    .map(|row| row.iter().map(|&v| v != U::zero()).collect())
                    .collect()
            },
        ));

        self
    }

    /// Adds eccentricity information to the graph and returns a new `Graph` instance.
    ///
    /// The eccentricity is calculated as the length of the frontier sizes starting from each cluster.
    ///
    /// # Returns
    ///
    /// A new `Graph` instance with added eccentricity information.
    #[must_use]
    pub fn with_eccentricities(mut self) -> Self {
        self.frontier_sizes = Some(
            self.clusters
                .iter()
                .map(|&c| {
                    let eccentricity = self
                        .traverse(c)
                        .unwrap_or_else(|_| unreachable!("Clusters are guaranteed to be in the graph"));
                    (c, eccentricity.1)
                })
                .collect(),
        );

        self
    }

    /// Finds and returns connected component clusters within the graph.
    ///
    /// This method identifies clusters in the graph that are part of separate connected components
    /// and returns a vector of sets where each set contains clusters from one connected component.
    ///
    /// # Returns
    ///
    /// A vector of sets where each set represents a connected component of clusters,
    ///
    #[must_use]
    pub fn find_component_clusters(&'a self) -> Vec<VertexSet<'a, U>> {
        let mut components = Vec::new();

        let mut unvisited = self.clusters.clone();
        while !unvisited.is_empty() {
            let &start = unvisited
                .iter()
                .next()
                .unwrap_or_else(|| (unreachable!("loop guarantees unvisited is not empty")));
            let (visited, _) = self
                .traverse(start)
                .unwrap_or_else(|_| (unreachable!("Clusters are guaranteed to be in the graph")));

            unvisited.retain(|&c| !visited.contains(c));

            // TODO: Also grab adjacency map, distance matrix, and adjacency matrix
            components.push(visited);
        }
        components
    }

    /// Returns a reference to the set of clusters in the graph.
    ///
    /// This method returns a reference to the set of clusters contained within the graph.
    ///
    /// # Returns
    ///
    /// A reference to the set of edges in the graph.
    #[must_use]
    pub const fn edges(&self) -> &EdgeSet<'a, U> {
        &self.edges
    }

    /// Returns the number of clusters in the graph, also known as the vertex cardinality.
    ///
    /// This method calculates and returns the total number of clusters in the graph.
    ///
    /// # Returns
    ///
    /// The number of clusters in the graph.
    #[must_use]
    pub fn vertex_cardinality(&self) -> usize {
        self.clusters.len()
    }

    /// Returns the number of edges in the graph, also known as the edge cardinality.
    ///
    /// This method calculates and returns the total number of edges in the graph.
    ///
    /// # Returns
    ///
    /// The number of edges in the graph.
    #[must_use]
    pub fn edge_cardinality(&self) -> usize {
        self.edges.len()
    }

    /// Returns the total population represented by the clusters in the graph.
    ///
    /// This method calculates and returns the total population represented by all clusters in the graph.
    ///
    /// # Returns
    ///
    /// The total population represented by the clusters in the graph.
    #[must_use]
    pub const fn population(&self) -> usize {
        self.population
    }

    /// Returns the minimum depth in the hierarchy of clusters in the graph.
    ///
    /// This method returns the minimum depth in the hierarchy of clusters in the graph. The depth represents
    /// the level or distance from the root cluster in the hierarchical structure of the graph.
    ///
    /// # Returns
    ///
    /// The minimum depth in the hierarchy of clusters.
    #[must_use]
    pub const fn min_depth(&self) -> usize {
        self.min_depth
    }

    /// Returns the maximum depth in the hierarchy of clusters in the graph.
    ///
    /// This method returns the maximum depth in the hierarchy of clusters in the graph. The depth represents
    /// the level or distance from the root cluster in the hierarchical structure of the graph.
    ///
    /// # Returns
    ///
    /// The maximum depth in the hierarchy of clusters.
    #[must_use]
    pub const fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the range of depths in the hierarchy of clusters in the graph as a tuple (`min_depth`, `max_depth`).
    ///
    /// This method returns the range of depths in the hierarchy of clusters in the graph. The depth range
    /// is represented as a tuple where the first element (`min_depth`) is the minimum depth, and the second
    /// element (`max_depth`) is the maximum depth.
    ///
    /// # Returns
    ///
    /// A tuple representing the depth range with `min_depth` as the first element and `max_depth` as the second element.
    ///
    #[must_use]
    pub const fn depth_range(&self) -> (usize, usize) {
        (self.min_depth, self.max_depth)
    }

    /// Returns a reference to the adjacency map representing adjacency relationships between clusters.
    ///
    /// This method returns a reference to the adjacency map that represents the adjacency relationships
    /// between clusters in the graph.
    ///
    /// # Returns
    ///
    /// A reference to the adjacency map.
    #[must_use]
    pub const fn adjacency_map(&'a self) -> &AdjacencyMap<U> {
        &self.adjacency_map
    }

    /// Returns an ordered slice of references to clusters based on the order of insertion into the graph.
    ///
    /// This method returns a slice of references to clusters in the order they were inserted into the graph.
    ///
    /// # Returns
    ///
    /// A slice of references to clusters in their insertion order.
    #[must_use]
    pub fn ordered_clusters(&self) -> &[&Vertex<U>] {
        &self.ordered_clusters
    }

    /// Returns a reference to the distance matrix of the graph.
    ///
    /// This method returns a reference to the distance matrix of the graph. The distance matrix
    /// should be computed by calling the `with_distance_matrix` method before using this function.
    ///
    /// # Returns
    ///
    /// An `Option` containing:
    /// - `Some` with a reference to the distance matrix, which is a 2D vector of the specified number type,
    ///   if `with_distance_matrix` was called before using this method.
    /// - `None` if the distance matrix is not available, indicating that `with_distance_matrix`
    ///   should be called before using this method.
    ///
    #[must_use]
    pub fn distance_matrix(&self) -> Option<&[Vec<U>]> {
        self.distance_matrix.as_deref()
    }

    /// Returns a reference to the adjacency matrix of the graph.
    ///
    /// This method returns a reference to the adjacency matrix of the graph. The adjacency matrix
    /// should be computed by calling the `with_adjacency_matrix` method before using this function.
    ///
    /// # Returns
    ///
    /// An `Option` where:
    /// - `Some` contains a reference to the adjacency matrix, which is a 2D vector of booleans.
    /// - `None` indicates that `with_adjacency_matrix` was not called before using this method.
    ///
    #[must_use]
    pub fn adjacency_matrix(&self) -> Option<&[Vec<bool>]> {
        self.adjacency_matrix.as_deref()
    }

    /// Calculates and returns the diameter of the graph.
    ///
    /// The diameter is the maximum eccentricity value among all clusters in the graph.
    /// It represents the longest shortest path between any two clusters in the graph.
    ///
    /// # Returns
    ///
    /// The diameter of the graph as a `usize` value.
    ///
    /// # Errors
    ///
    /// Returns an error if there is an issue computing eccentricity for any cluster, or if there are no clusters in the graph.
    pub fn diameter(&'a self) -> Result<usize, String> {
        self.clusters
            .iter()
            .map(|&c| {
                self.eccentricity(c)
                    .map_err(|e| format!("Error computing eccentricity: {e}"))
            })
            .max()
            .ok_or_else(|| "No clusters in the graph".to_string())?
    }

    /// Asserts whether a given cluster is contained within the graph.
    ///
    /// # Arguments
    ///
    /// * `c`: The cluster to check for containment in the graph.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the cluster is contained in the graph. Returns an `Err` containing an error message
    /// if the cluster is not in the graph.
    ///
    /// # Errors
    ///
    /// An error is returned when the specified cluster is not present in the graph.
    fn assert_contains(&self, c: &Vertex<U>) -> Result<(), String> {
        if self.clusters.contains(&c) {
            Ok(())
        } else {
            Err(format!("Cluster {c} is not in this graph."))
        }
    }

    /// Returns the vertex degree (number of neighbors) of a given cluster.
    ///
    /// # Arguments
    ///
    /// * `c`: The cluster for which the vertex degree is calculated.
    ///
    /// # Returns
    ///
    /// The vertex degree of the specified cluster.
    ///
    /// # Errors
    ///
    /// If the specified cluster is not present in the graph.
    pub fn vertex_degree(&'a self, c: &Vertex<U>) -> Result<usize, String> {
        match self.neighbors_of(c) {
            Ok(neighbors) => Ok(neighbors.len()),
            Err(e) => Err(e),
        }
    }

    /// Returns a reference to the set of neighbors of a given cluster.
    ///
    /// # Arguments
    ///
    /// * `c`: The cluster for which neighbors are retrieved.
    ///
    /// # Returns
    ///
    /// A reference to the set of neighbor clusters.
    ///
    /// # Errors
    ///
    /// Returns an error if the given cluster is not present in the graph.
    pub fn neighbors_of(&'a self, c: &Vertex<U>) -> Result<&VertexSet<U>, String> {
        self.adjacency_map
            .get(c)
            .ok_or_else(|| format!("Cluster {c} not found in adjacency_map"))
    }

    /// Performs an unchecked traverse of the graph starting from the given cluster and returns visited clusters and frontier sizes.
    ///
    /// # Arguments
    ///
    /// * `start`: The cluster from which the graph traversal starts.
    ///
    /// # Returns
    ///
    /// A tuple of visited clusters and a vector of frontier sizes.
    ///
    /// # Errors
    ///
    /// None.
    ///
    /// # Panics
    ///
    /// * If the start cluster is not present in the graph.
    pub fn traverse(&'a self, start: &'a Vertex<U>) -> Result<(VertexSet<U>, Vec<usize>), String> {
        self.assert_contains(start)?;

        let mut visited: HashSet<&Vertex<U>> = HashSet::new();
        let mut frontier: HashSet<&Vertex<U>> = HashSet::new();
        frontier.insert(start);
        let mut frontier_sizes: Vec<usize> = Vec::new();

        while !frontier.is_empty() {
            visited.extend(frontier.iter().copied());
            frontier = frontier
                .iter()
                .flat_map(|&c| {
                    self.neighbors_of(c)
                        .unwrap_or_else(|_| unreachable!("We asserted cluster is in the graph"))
                })
                .filter(|&n| !((visited.contains(n)) || (frontier.contains(n))))
                .copied()
                .collect();
            frontier_sizes.push(frontier.len());
        }

        Ok((visited, frontier_sizes))
    }

    /// Retrieves the frontier sizes for a specified cluster.
    ///
    /// Frontier sizes indicate the number of clusters encountered at each level of traversal
    /// starting from the given cluster. The results are provided as a reference to a slice.
    ///
    /// # Arguments
    ///
    /// - `c`: A reference to the cluster for which frontier sizes are to be retrieved.
    ///
    /// # Returns
    ///
    /// - `Ok(frontier_sizes)`: A reference to a slice containing the frontier sizes.
    ///
    /// # Errors
    ///
    /// If the specified cluster is not part of the graph, an error message is returned.
    ///
    pub fn frontier_sizes(&'a self, c: &'a Vertex<U>) -> Result<&[usize], String> {
        self.assert_contains(c)?;

        Ok(self.frontier_sizes.as_ref().map_or_else(
            || Err("Please call with_eccentricities before using this method".to_string()),
            |sizes| {
                sizes
                    .get(c)
                    .ok_or_else(|| unreachable!("We asserted cluster is in the graph"))
            },
        )?)
    }

    /// Calculates the eccentricity of a specified cluster.
    ///
    /// The eccentricity represents the length of the frontier sizes starting from the given cluster.
    ///
    /// # Arguments
    ///
    /// - `c`: A reference to the cluster for which eccentricity is to be calculated.
    ///
    /// # Returns
    ///
    /// - `Ok(eccentricity)`: The calculated eccentricity as a `usize` value.
    ///
    /// # Errors
    ///
    /// If the specified cluster is not part of the graph or if `with_eccentricities` was not called, an error message is returned.
    pub fn eccentricity(&'a self, c: &'a Vertex<U>) -> Result<usize, String> {
        self.frontier_sizes(c).map_or_else(
            |_| Err("Please call with_eccentricities before using this method".to_string()),
            |frontier_sizes| Ok(frontier_sizes.len()),
        )
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::collections::HashSet;

    use crate::{chaoda::pretrained_models, Cluster, PartitionCriteria, Tree, VecDataset};
    use distances::number::Float;
    use distances::Number;
    use rand::SeedableRng;

    use super::*;

    /// Generate a dataset with the given cardinality and dimensionality.
    pub fn gen_dataset(
        cardinality: usize,
        dimensionality: usize,
        seed: u64,
        metric: fn(&Vec<f32>, &Vec<f32>) -> f32,
    ) -> VecDataset<Vec<f32>, f32, usize> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data = symagen::random_data::random_tabular(cardinality, dimensionality, -1., 1., &mut rng);
        let name = "test".to_string();
        VecDataset::new(name, data, metric, false)
    }

    /// Euclidean distance between two vectors.
    #[allow(clippy::ptr_arg)]
    pub fn euclidean<T: Number, F: Float>(x: &Vec<T>, y: &Vec<T>) -> F {
        distances::vectors::euclidean(x, y)
    }

    #[test]
    fn test_graph() {
        let cardinality = 1000;
        let data = gen_dataset(cardinality, 10, 42, euclidean);

        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
        let raw_tree = Tree::new(data, Some(42)).partition(&partition_criteria, Some(42));
        for i in 4..raw_tree.depth() {
            let selected_clusters = select_clusters(
                raw_tree.root(),
                &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
                i,
            )
            .unwrap();

            let edges = detect_edges(&selected_clusters, raw_tree.data());

            let graph = Graph::from_clusters_and_edges(selected_clusters.clone(), edges.clone());
            assert!(graph.is_ok());
            if let Ok(graph) = graph {
                let graph = graph.with_adjacency_matrix().with_distance_matrix();
                assert_eq!(graph.population(), cardinality);
                test_properties(&graph, &selected_clusters, &edges);
                test_adjacency_map(&graph);
                test_matrix(&graph);
            }
        }
    }

    #[test]
    fn create_graph_from_tree() {
        let data = gen_dataset(1000, 10, 42, euclidean);
        let partition_criteria: PartitionCriteria<f32> = PartitionCriteria::default();
        let raw_tree = Tree::new(data, Some(42))
            .partition(&partition_criteria, Some(42))
            .normalize_ratios();

        let graph = Graph::from_tree(
            &raw_tree,
            &pretrained_models::get_meta_ml_scorers().first().unwrap().1,
            4,
        );
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        let graph = graph.with_adjacency_matrix().with_distance_matrix();

        test_adjacency_map(&graph);
        test_matrix(&graph);
    }

    fn test_properties(graph: &Graph<f32>, selected_clusters: &HashSet<&Vertex<f32>>, edges: &HashSet<Edge<f32>>) {
        // assert edges and clusters are correct
        assert_eq!(graph.clusters.len(), selected_clusters.len());
        assert_eq!(graph.edges().len(), edges.len());

        let reference_population = selected_clusters.iter().fold(0, |acc, &c| acc + c.cardinality());
        assert_eq!(graph.population(), reference_population);
        let components = graph.find_component_clusters();

        graph.clusters.iter().for_each(|c1| {
            assert!(graph.ordered_clusters.contains(c1));
        });

        // assert ordered clusters are in correct order
        for i in 1..graph.ordered_clusters().len() {
            assert!(graph.ordered_clusters().get(i) > graph.ordered_clusters().get(i - 1));
        }

        let num_clusters_in_components = components.iter().map(std::collections::HashSet::len).sum::<usize>();
        assert_eq!(num_clusters_in_components, selected_clusters.len());

        // assert the number of clusters in a component is equal to the number of clusters in each of its cluster's traversals
        for component in &components {
            for c in component {
                if let Ok(traversal_result) = graph.traverse(c) {
                    assert_eq!(traversal_result.0.len(), component.len());
                    // assert_eq!(traversal_result.1.len(), component.len());
                }
            }
        }
    }

    fn test_adjacency_map(graph: &Graph<f32>) {
        let adj_map = graph.adjacency_map();
        assert_eq!(adj_map.len(), graph.clusters.len());

        for component in &graph.find_component_clusters() {
            for c in component {
                let adj = adj_map.get(c).unwrap();
                for adj_c in adj {
                    assert!(component.contains(adj_c));
                }
            }
        }
    }

    fn test_matrix(graph: &Graph<f32>) {
        assert_eq!(graph.adjacency_map.len(), graph.distance_matrix().unwrap().len());
        assert_eq!(graph.adjacency_map.len(), graph.adjacency_matrix().unwrap().len());

        assert_eq!(
            graph.distance_matrix().unwrap().first().unwrap().len(),
            graph.adjacency_matrix().unwrap().first().unwrap().len()
        );

        assert!(distance_mat_is_symmetric(graph));

        let flat_adj_mat = graph.adjacency_matrix().unwrap().iter().flatten();
        let flat_dist_mat = graph.distance_matrix().unwrap().iter().flatten();

        for (&has_edge, &distance) in flat_adj_mat.zip(flat_dist_mat) {
            if has_edge {
                // possibly false fails if distance is actually 0?
                assert!(!float_cmp::approx_eq!(f32, distance, 0.));
            } else {
                assert!(float_cmp::approx_eq!(f32, distance, 0.));
            }
        }
    }

    fn distance_mat_is_symmetric(graph: &Graph<f32>) -> bool {
        let matrix = graph.distance_matrix().unwrap();
        let num_rows = matrix.len();
        let num_cols = matrix[0].len();

        if num_rows != num_cols {
            return false; // Non-square matrix cannot be symmetric
        }

        for i in 0..num_rows {
            for j in 0..num_cols {
                if (matrix[i][j] - matrix[j][i]).abs() > 1e-6 {
                    return false; // Elements at (i, j) and (j, i) are not equal, matrix is not symmetric
                }
            }
        }

        true
    }
}
