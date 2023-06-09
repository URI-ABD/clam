use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::Hasher;

use super::cluster::Cluster;
use super::dataset::Dataset;
use super::number::Number;

type ClusterSet<'a, T, D> = HashSet<&'a Cluster<'a, T, D>>;
type EdgeSet<'a, T, D> = HashSet<&'a Edge<'a, T, D>>;
type AdjacencyMap<'a, T, D> = HashMap<&'a Cluster<'a, T, D>, ClusterSet<'a, T, D>>;
type FrontierSizes<'a, T, D> = HashMap<&'a Cluster<'a, T, D>, Vec<usize>>;

/// Two `Cluster`s have an `Edge` between them if they have overlapping volumes.
///
/// In CLAM, all `Edge`s are bi-directional.
#[derive(Debug, Clone)]
pub struct Edge<'a, T: Number, D: Dataset<T, f64>> {
    left: &'a Cluster<'a, T, D>,
    right: &'a Cluster<'a, T, D>,
    distance: f64,
}

impl<'a, T: Number, D: Dataset<T, f64>> PartialEq for Edge<'a, T, D> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

/// Two `Edge`s are equal if they connect the same two `Cluster`s.
impl<'a, T: Number, D: Dataset<T, f64>> Eq for Edge<'a, T, D> {}

impl<'a, T: Number, D: Dataset<T, f64>> std::fmt::Display for Edge<'a, T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:} -- {:}", self.left, self.right)
    }
}

impl<'a, T: Number, D: Dataset<T, f64>> Hash for Edge<'a, T, D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{self}").hash(state)
    }
}

impl<'a, T: Number, D: Dataset<T, f64>> Edge<'a, T, D> {
    /// Creates a new `Edge` from the given `Cluster`s and the distance between
    /// them.
    ///
    /// It is upon the user to verify that the two `Cluster`s are close enough
    /// to have an edge between them.
    pub fn new(left: &'a Cluster<'a, T, D>, right: &'a Cluster<'a, T, D>, distance: f64) -> Self {
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

    /// Whether this edge has the given `Cluster` at one of its ends.
    pub fn contains(&self, c: &Cluster<'a, T, D>) -> bool {
        c == self.left || c == self.right
    }

    /// A 2-slice of the `Cluster`s in this `Edge`.
    pub fn clusters(&self) -> [&Cluster<'a, T, D>; 2] {
        [self.left, self.right]
    }

    /// A reference to the `Cluster` at the `left` end of the `Edge`.
    pub fn left(&self) -> &Cluster<'a, T, D> {
        self.left
    }

    /// A reference to the `Cluster` at the `right` end of the `Edge`.
    pub fn right(&self) -> &Cluster<'a, T, D> {
        self.right
    }

    /// The distance between the two `Cluster`s connected by this `Edge`.
    pub fn distance(&self) -> f64 {
        self.distance
    }

    /// Whether this is an edge from a `Cluster` to itself.
    pub fn is_circular(&self) -> bool {
        self.left == self.right
    }

    /// Returns the neighbor of the given `Cluster` in this `Edge`.
    ///
    /// Err:
    ///
    /// * If `c` is not one of the `Cluster`s connected by this `Edge`.
    pub fn neighbor(&self, c: &Cluster<'a, T, D>) -> Result<&Cluster<'a, T, D>, String> {
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
pub struct Graph<'a, T: Number, D: Dataset<T, f64>> {
    clusters: ClusterSet<'a, T, D>,
    edges: EdgeSet<'a, T, D>,
    adjacency_map: AdjacencyMap<'a, T, D>,
    population: usize,
    min_depth: usize,
    max_depth: usize,
    ordered_clusters: Vec<&'a Cluster<'a, T, D>>,
    distance_matrix: Option<Vec<Vec<f64>>>,
    adjacency_matrix: Option<Vec<Vec<bool>>>,
    frontier_sizes: Option<FrontierSizes<'a, T, D>>, // TODO: Bench when replacing with DashMap
}

impl<'a, T: Number, D: Dataset<T, f64>> Graph<'a, T, D> {
    /// Create a new `Graph` from the given `clusters` and `edges`. The easiest
    /// and most efficient way to construct a graph is from methods in
    /// `Manifold`.
    ///
    /// # Arguments:
    ///
    /// * `clusters`: The set of `Cluster`s with which to build the `Graph`.
    /// * `edges`: The set of `Edge`s with which to build the `Graph`.
    pub fn new(clusters: ClusterSet<'a, T, D>, edges: EdgeSet<'a, T, D>) -> Self {
        assert!(!clusters.is_empty());

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
            let mut adjacency_map: AdjacencyMap<T, D> = clusters.iter().map(|&c| (c, HashSet::new())).collect();
            edges.iter().for_each(|&e| {
                adjacency_map.get_mut(e.left()).unwrap().insert(e.right());
                adjacency_map.get_mut(e.right()).unwrap().insert(e.left());
            });
            adjacency_map
        };

        Self {
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
        }
    }

    fn compute_distance_matrix(&self) -> Vec<Vec<f64>> {
        let indices = self
            .ordered_clusters
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect::<HashMap<_, _>>();
        let mut matrix = vec![vec![0.; self.vertex_cardinality()]; self.vertex_cardinality()];
        self.edges.iter().for_each(|&e| {
            let i = *indices.get(e.left()).unwrap();
            let j = *indices.get(e.right()).unwrap();
            matrix[i][j] = e.distance();
            matrix[j][i] = e.distance();
        });
        matrix
    }

    /// Computes the distance matrix for the `Graph` and stores it as an
    /// internal property.
    pub fn with_distance_matrix(mut self) -> Self {
        self.distance_matrix = Some(self.compute_distance_matrix());
        self
    }

    /// Computes the adjacency matrix for the `Graph` and stores it as an
    /// internal property.
    ///
    /// # Panics:
    ///
    /// * If called before calling `with_distance_matrix`.
    pub fn with_adjacency_matrix(mut self) -> Self {
        self.adjacency_matrix = Some(
            self.distance_matrix()
                .iter()
                .map(|row| row.iter().map(|v| *v != 0.).collect())
                .collect(),
        );
        self
    }

    pub fn with_eccentricities(&'a self) -> Self {
        let frontier_sizes = Some(
            self.clusters
                .iter()
                .map(|&c| (c, self.unchecked_traverse(c).1))
                .collect(),
        );

        Self {
            clusters: self.clusters.clone(),
            edges: self.edges.clone(),
            adjacency_map: self.adjacency_map.clone(),
            population: self.population,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
            ordered_clusters: self.ordered_clusters.clone(),
            distance_matrix: self.distance_matrix.clone(),
            adjacency_matrix: self.adjacency_matrix.clone(),
            frontier_sizes,
        }
    }

    #[allow(clippy::manual_retain)]
    pub fn find_component_clusters(&'a self) -> Vec<ClusterSet<'a, T, D>> {
        let mut components = Vec::new();

        let mut unvisited = self.clusters.clone();
        while !unvisited.is_empty() {
            let &start = unvisited.iter().next().unwrap();
            let (visited, _) = self.unchecked_traverse(start);

            // TODO: bench this using `unvisited.retain(|c| !visited.contains(c))`
            unvisited = unvisited.into_iter().filter(|&c| !visited.contains(c)).collect();

            // TODO: Also grab adjacency map, distance matrix, and adjacency matrix
            components.push(visited);
        }

        components
    }

    pub fn clusters(&self) -> &ClusterSet<'a, T, D> {
        &self.clusters
    }

    pub fn edges(&self) -> &EdgeSet<'a, T, D> {
        &self.edges
    }

    pub fn vertex_cardinality(&self) -> usize {
        self.clusters.len()
    }

    pub fn edge_cardinality(&self) -> usize {
        self.edges.len()
    }

    pub fn population(&self) -> usize {
        self.population
    }

    pub fn min_depth(&self) -> usize {
        self.min_depth
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn depth_range(&self) -> (usize, usize) {
        (self.min_depth, self.max_depth)
    }

    pub fn adjacency_map(&'a self) -> &AdjacencyMap<T, D> {
        &self.adjacency_map
    }

    pub fn ordered_clusters(&self) -> &[&Cluster<'a, T, D>] {
        &self.ordered_clusters
    }

    pub fn distance_matrix(&self) -> &Vec<Vec<f64>> {
        self.distance_matrix
            .as_ref()
            .expect("Please call `with_distance_matrix` on the Graph before using `distance_matrix`.")
    }

    pub fn adjacency_matrix(&self) -> &[Vec<bool>] {
        self.adjacency_matrix
            .as_ref()
            .expect("Please call `with_adjacency_matrix` on the Graph before using `adjacency_matrix`.")
    }

    pub fn diameter(&'a self) -> usize {
        self.clusters
            .iter()
            .map(|&c| self.unchecked_eccentricity(c))
            .max()
            .unwrap()
    }

    fn assert_contains(&self, c: &Cluster<'a, T, D>) -> Result<(), String> {
        if self.clusters.contains(&c) {
            Ok(())
        } else {
            Err(format!("Cluster {c} is not in this graph."))
        }
    }

    pub fn unchecked_vertex_degree(&'a self, c: &Cluster<'a, T, D>) -> usize {
        self.unchecked_neighbors_of(c).len()
    }

    pub fn vertex_degree(&'a self, c: &Cluster<'a, T, D>) -> Result<usize, String> {
        self.assert_contains(c)?;
        Ok(self.unchecked_vertex_degree(c))
    }

    pub fn unchecked_neighbors_of(&'a self, c: &Cluster<'a, T, D>) -> &ClusterSet<T, D> {
        self.adjacency_map.get(c).unwrap()
    }

    pub fn neighbors_of(&'a self, c: &Cluster<'a, T, D>) -> Result<&ClusterSet<T, D>, String> {
        self.assert_contains(c)?;
        Ok(self.unchecked_neighbors_of(c))
    }

    pub fn unchecked_traverse(&'a self, start: &'a Cluster<'a, T, D>) -> (ClusterSet<T, D>, Vec<usize>) {
        let mut visited: HashSet<&Cluster<T, D>> = HashSet::new();
        let mut frontier: HashSet<&Cluster<T, D>> = HashSet::new();
        frontier.insert(start);
        let mut frontier_sizes: Vec<usize> = Vec::new();

        while !frontier.is_empty() {
            visited.extend(frontier.iter().copied());
            frontier = frontier
                .iter()
                .flat_map(|&c| self.unchecked_neighbors_of(c))
                .filter(|&n| !((visited.contains(n)) || (frontier.contains(n))))
                .copied()
                .collect();
            frontier_sizes.push(frontier.len());
        }

        (visited, frontier_sizes)
    }

    #[allow(clippy::type_complexity)]
    pub fn traverse(&'a self, start: &'a Cluster<'a, T, D>) -> Result<(ClusterSet<T, D>, Vec<usize>), String> {
        self.assert_contains(start)?;
        Ok(self.unchecked_traverse(start))
    }

    pub fn unchecked_frontier_sizes(&'a self, c: &'a Cluster<'a, T, D>) -> &[usize] {
        self.frontier_sizes
            .as_ref()
            .expect("Please call `with_eccentricities` before using this method.")
            .get(c)
            .unwrap()
    }

    pub fn frontier_sizes(&'a self, c: &'a Cluster<'a, T, D>) -> Result<&[usize], String> {
        self.assert_contains(c)?;
        Ok(self.unchecked_frontier_sizes(c))
    }

    pub fn unchecked_eccentricity(&'a self, c: &'a Cluster<'a, T, D>) -> usize {
        self.unchecked_frontier_sizes(c).len()
    }

    pub fn eccentricity(&'a self, c: &'a Cluster<'a, T, D>) -> Result<usize, String> {
        self.assert_contains(c)?;
        Ok(self.unchecked_eccentricity(c))
    }
}
