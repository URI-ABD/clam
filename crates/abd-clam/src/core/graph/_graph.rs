use core::hash::{Hash, Hasher};

use std::collections::{HashMap, HashSet};

use distances::Number;

// use crate::core::{cluster::Cluster, dataset::VecDataset};
use crate::core::cluster::Cluster;

/// A set of clusters with references to clusters in a graph.
pub type ClusterSet<'a, U> = HashSet<&'a Cluster<U>>;

/// A set of edges with references to edges in a graph.
pub type EdgeSet<'a, U> = HashSet<&'a Edge<'a, U>>;

/// A map that represents the adjacency relationship between clusters.
pub type AdjacencyMap<'a, U> = HashMap<&'a Cluster<U>, ClusterSet<'a, U>>;

/// A map that associates clusters with lists of frontier sizes.
pub type FrontierSizes<'a, U> = HashMap<&'a Cluster<U>, Vec<usize>>;

/// Two `Cluster`s have an `Edge` between them if they have overlapping volumes.
///
/// In CLAM, all `Edge`s are bi-directional.
#[derive(Debug, Clone)]
pub struct Edge<'a, U: Number> {
    /// A reference to the first `Cluster` connected by this `Edge`.
    left: &'a Cluster<U>,
    /// A reference to the second `Cluster` connected by this `Edge`.
    right: &'a Cluster<U>,

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
    pub fn new(left: &'a Cluster<U>, right: &'a Cluster<U>, distance: U) -> Self {
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
    pub fn contains(&self, c: &Cluster<U>) -> bool {
        c == self.left || c == self.right
    }

    /// Returns a 2-slice containing the `Cluster`s at the two ends of this `Edge`.
    ///
    /// # Returns
    ///
    /// A 2-slice containing the `Cluster`s at the left and right ends of the edge.
    pub const fn clusters(&self) -> [&Cluster<U>; 2] {
        [self.left, self.right]
    }

    /// Retrieves a reference to the `Cluster` at the `left` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the left end of the `Edge`.
    pub const fn left(&self) -> &Cluster<U> {
        self.left
    }

    /// Retrieves a reference to the `Cluster` at the `right` end of the `Edge`.
    ///
    /// # Returns
    ///
    /// A reference to the `Cluster` at the right end of the `Edge`.
    pub const fn right(&self) -> &Cluster<U> {
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
    pub fn neighbor(&self, c: &Cluster<U>) -> Result<&Cluster<U>, String> {
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
    clusters: ClusterSet<'a, U>,
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
    ordered_clusters: Vec<&'a Cluster<U>>,
    /// A distance matrix representing distances between clusters.
    distance_matrix: Option<Vec<Vec<U>>>,
    /// An adjacency matrix representing adjacency relationships between clusters.
    adjacency_matrix: Option<Vec<Vec<bool>>>,
    /// A mapping of clusters to their respective frontier sizes.
    #[allow(dead_code)]
    frontier_sizes: Option<FrontierSizes<'a, U>>, // TODO: Bench when replacing with DashMap
}

impl<'a, U: Number> Graph<'a, U> {
    /// Create a new `Graph` from the given `clusters` and `edges`. The easiest
    /// and most efficient way to construct a graph is from methods in `Manifold`.
    ///
    /// # Arguments
    ///
    /// * `clusters`: The set of `Cluster`s with which to build the `Graph`.
    /// * `edges`: The set of `Edge`s with which to build the `Graph`.
    ///
    /// # Returns
    ///
    /// Returns a `Result` with a new `Graph` instance constructed from the provided clusters and edges
    /// if the operation succeeds. Otherwise, returns an `Err` containing an error message.
    ///
    /// # Errors
    ///
    /// Returns an error if the provided `clusters` set is empty, indicating that a graph cannot be
    /// created with no clusters. Also returns an error if an edge refers to a cluster that is not in
    /// the `clusters` set.
    pub fn new(clusters: ClusterSet<'a, U>, edges: EdgeSet<'a, U>) -> Result<Self, String> {
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
            for &e in &edges {
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

        Ok(Self {
            ordered_clusters: clusters.iter().copied().collect(),
            clusters,
            edges,
            adjacency_map,
            population,
            min_depth,
            max_depth,
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
        self.edges.iter().for_each(|&e| {
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
    /// internal property.
    ///
    /// # Returns
    ///
    /// A new `Graph` instance with the updated adjacency matrix.
    ///
    /// # Panics
    ///
    /// This method panics if called before `with_distance_matrix`.
    #[must_use]
    pub fn with_adjacency_matrix(mut self) -> Self {
        self.adjacency_matrix = match self.distance_matrix() {
            Ok(distance_matrix) => Some(
                distance_matrix
                    .iter()
                    .map(|row| row.iter().map(|&v| v == U::zero()).collect())
                    .collect(),
            ),
            Err(err) => {
                format!("Error getting distance_matrix: {err}");
                None
            }
        };
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
    /// A `Result` where `Ok` contains a vector of sets where each set represents a connected component of clusters,
    /// and `Err` contains an error message if there's an issue during the computation.
    #[must_use]
    pub fn find_component_clusters(&'a self) -> Vec<ClusterSet<'a, U>> {
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
    /// A reference to the set of clusters in the graph.
    #[must_use]
    pub const fn clusters(&self) -> &ClusterSet<'a, U> {
        &self.clusters
    }

    /// Returns a reference to the set of edges in the graph.
    ///
    /// This method returns a reference to the set of edges representing connections between clusters
    /// in the graph.
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
    pub fn ordered_clusters(&self) -> &[&Cluster<U>] {
        &self.ordered_clusters
    }

    /// Returns a reference to the distance matrix of the graph.
    ///
    /// This method returns a reference to the distance matrix of the graph. The distance matrix
    /// should be computed by calling the `with_distance_matrix` method before using this function.
    ///
    /// # Returns
    ///
    /// A `Result` where:
    /// - `Ok` contains a reference to the distance matrix, which is a 2D vector of the specified number type.
    /// - `Err` contains an error message if `with_distance_matrix` was not called before using this method.
    ///
    /// # Errors
    ///
    /// Returns an error message if the distance matrix is not available, indicating that `with_distance_matrix`
    /// should be called before using this method.
    pub fn distance_matrix(&self) -> Result<&[Vec<U>], String> {
        Ok(self.distance_matrix.as_ref().ok_or_else(|| {
            "Please call `with_distance_matrix` on the Graph before using `distance_matrix`.".to_string()
        })?)
    }

    /// Returns a reference to the adjacency matrix of the graph.
    ///
    /// This method returns a reference to the adjacency matrix of the graph. The adjacency matrix
    /// should be computed by calling the `with_adjacency_matrix` method before using this function.
    ///
    /// # Returns
    ///
    /// A `Result` where:
    /// - `Ok` contains a reference to the adjacency matrix, which is a 2D vector of booleans.
    /// - `Err` contains an error message if `with_adjacency_matrix` was not called before using this method.
    ///
    /// # Errors
    ///
    /// Returns an error message if the adjacency matrix is not available, indicating that `with_adjacency_matrix`
    /// should be called before using this method.
    pub fn adjacency_matrix(&self) -> Result<&[Vec<bool>], String> {
        Ok(self.adjacency_matrix.as_ref().ok_or_else(|| {
            "Please call `with_adjacency_matrix` on the Graph before using `adjacency_matrix`.".to_string()
        })?)
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
    fn assert_contains(&self, c: &Cluster<U>) -> Result<(), String> {
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
    /// * `c`: The cluster for which you want to determine the vertex degree.
    ///
    /// # Panics
    ///
    /// * If the specified cluster is not present in the graph.
    // pub fn unchecked_vertex_degree(&'a self, c: &Cluster<U>) -> usize {
    //     self.unchecked_neighbors_of(c).len()
    // }

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
    pub fn vertex_degree(&'a self, c: &Cluster<U>) -> Result<usize, String> {
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
    /// # Panics
    ///
    /// Panics if the specified cluster is not present in the graph.
    // pub fn unchecked_neighbors_of(&'a self, c: &Cluster<U>) -> Result<&ClusterSet<U>, String> {
    //     self.adjacency_map
    //         .get(c)
    //         .ok_or_else(|| format!("Cluster {} not found in adjacency_map", c))
    // }

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
    pub fn neighbors_of(&'a self, c: &Cluster<U>) -> Result<&ClusterSet<U>, String> {
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
    pub fn traverse(&'a self, start: &'a Cluster<U>) -> Result<(ClusterSet<U>, Vec<usize>), String> {
        self.assert_contains(start)?;

        let mut visited: HashSet<&Cluster<U>> = HashSet::new();
        let mut frontier: HashSet<&Cluster<U>> = HashSet::new();
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

    /// Traverses the graph starting from the given cluster and returns visited clusters and frontier sizes as a `Result`.
    ///
    /// # Arguments
    ///
    /// * `start`: The cluster from which the graph traversal starts.
    ///
    /// # Returns
    ///
    /// A `Result` containing a tuple of visited clusters and a vector of frontier sizes.
    ///
    /// # Errors
    ///
    /// If the given start cluster is not part of the graph.
    // #[allow(clippy::type_complexity)]
    // pub fn traverse(&'a self, start: &'a Cluster<U>) -> Result<(ClusterSet<U>, Vec<usize>), String> {
    //     self.assert_contains(start)?;
    //     Ok(self.unchecked_traverse(start))
    // }

    /// Returns the frontier sizes for a given cluster.
    ///
    /// The frontier sizes represent the number of clusters encountered at each level of traversal
    /// starting from the given cluster. It is calculated and returned as a reference.
    ///
    /// # Panics
    ///
    /// * If `with_eccentricities` is not called before using this method
    ///
    /// # Arguments
    ///
    /// * `c`: The cluster for which frontier sizes are calculated.
    ///
    /// # Returns
    ///
    /// A reference to a slice of frontier sizes.
    // pub const fn unchecked_frontier_sizes(&'a self, _c: &'a Cluster<U>) -> &[usize] {
    //     todo!()
    //
    //     // self.frontier_sizes
    //     //     .as_ref()
    //     //     .expect("Please call `with_eccentricities` before using this method.")
    //     //     .get(c)
    //     //     .unwrap()
    // }

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
    pub fn frontier_sizes(&'a self, c: &'a Cluster<U>) -> Result<&[usize], String> {
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

    /// Returns the eccentricity of a given cluster as a `Result`.
    ///
    /// The eccentricity is calculated as the length of the frontier sizes starting from the given cluster.
    ///
    /// Panics:
    ///
    /// * If the specified cluster is not present in the graph.

    // pub const fn unchecked_eccentricity(&'a self, c: &'a Cluster<U>) -> usize {
    //     self.unchecked_frontier_sizes(c).len()
    // }

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
    pub fn eccentricity(&'a self, c: &'a Cluster<U>) -> Result<usize, String> {
        self.frontier_sizes(c).map_or_else(
            |_| Err("Please call with_eccentricities before using this method".to_string()),
            |frontier_sizes| Ok(frontier_sizes.len()),
        )
    }
}
