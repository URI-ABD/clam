use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::RwLock;

use crate::prelude::*;

pub type ClusterSet<'a, T, U> = HashSet<&'a Cluster<'a, T, U>>;
pub type EdgeSet<'a, T, U> = HashSet<&'a Edge<'a, T, U>>;
pub type AdjacencyMap<'a, T, U> = HashMap<&'a Cluster<'a, T, U>, ClusterSet<'a, T, U>>;
pub type FrontierSizes<'a, T, U> = HashMap<&'a Cluster<'a, T, U>, Vec<usize>>;
// pub type GraphVec<'a, T, U> = Vec<Graph<'a, T, U>>;

#[derive(Debug, Clone)]
pub struct Edge<'a, T: Number, U: Number> {
    left: &'a Cluster<'a, T, U>,
    right: &'a Cluster<'a, T, U>,
    distance: U,
}

/// Two edges are the same if they connect the same two clusters.
impl<'a, T: Number, U: Number> PartialEq for Edge<'a, T, U> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

impl<'a, T: Number, U: Number> Eq for Edge<'a, T, U> {}

/// This may be used for creating dot files for graphviz, or for hashing an edge.
impl<'a, T: Number, U: Number> std::fmt::Display for Edge<'a, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:} -- {:}", self.left.name(), self.right.name())
    }
}

impl<'a, T: Number, U: Number> Hash for Edge<'a, T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:}", self).hash(state)
    }
}

impl<'a, T: Number, U: Number> Edge<'a, T, U> {
    pub fn new(left: &'a Cluster<T, U>, right: &'a Cluster<T, U>, distance: U) -> Self {
        if left < right {
            Edge { left, right, distance }
        } else {
            Edge {
                left: right,
                right: left,
                distance,
            }
        }
    }

    pub fn contains(&self, c: &Cluster<T, U>) -> bool {
        c == self.left || c == self.right
    }

    pub fn clusters(&self) -> (&Cluster<T, U>, &Cluster<T, U>) {
        (self.left(), self.right())
    }

    pub fn left(&self) -> &Cluster<T, U> {
        self.left
    }

    pub fn right(&self) -> &Cluster<T, U> {
        self.right
    }

    pub fn distance(&self) -> U {
        self.distance
    }

    pub fn is_to_self(&self) -> bool {
        self.left == self.right
    }

    pub fn neighbor(&self, c: &Cluster<T, U>) -> Result<&Cluster<T, U>, String> {
        if c == self.left {
            Ok(self.right())
        } else if c == self.right {
            Ok(self.left())
        } else {
            Err(format!("Cluster {:} is not in this edge {:}.", c, self))
        }
    }
}

#[derive(Debug, Clone)]
pub struct Graph<'a, T: Number, U: Number> {
    space: &'a dyn Space<T, U>,
    cluster_set: ClusterSet<'a, T, U>,
    edge_set: EdgeSet<'a, T, U>,
    adjacency_map: AdjacencyMap<'a, T, U>,
    population: usize,
    min_depth: usize,
    max_depth: usize,
    ordered_clusters: Vec<&'a Cluster<'a, T, U>>,
    ordered_edges: Vec<&'a Edge<'a, T, U>>,
    distance_matrix: Option<Vec<Vec<U>>>,
    adjacency_matrix: Option<Vec<Vec<bool>>>,
    components: Option<Vec<Graph<'a, T, U>>>,
    frontier_sizes: Arc<RwLock<FrontierSizes<'a, T, U>>>, // TODO: Bench when replacing with DashMap
}

impl<'a, T: Number, U: Number> Graph<'a, T, U> {
    pub fn new(clusters: ClusterSet<'a, T, U>, edges: EdgeSet<'a, T, U>) -> Self {
        assert!(!clusters.is_empty());

        let &c = clusters.iter().next().unwrap();
        let space = c.space();

        let population: usize = clusters.iter().map(|c| c.cardinality()).sum();
        let min_depth: usize = clusters.iter().map(|c| c.depth()).min().unwrap();
        let max_depth: usize = clusters.iter().map(|c| c.depth()).max().unwrap();

        let ordered_clusters: Vec<_> = clusters.iter().copied().collect();
        let ordered_edges: Vec<_> = edges.iter().copied().collect();

        let adjacency_map = {
            let mut adjacency_map: AdjacencyMap<T, U> = clusters.iter().map(|&c| (c, HashSet::new())).collect();
            edges.iter().for_each(|&e| {
                adjacency_map.get_mut(e.left()).unwrap().insert(e.right());
                adjacency_map.get_mut(e.right()).unwrap().insert(e.left());
            });
            adjacency_map
        };

        Graph {
            space,
            cluster_set: clusters,
            edge_set: edges,
            adjacency_map,
            population,
            min_depth,
            max_depth,
            ordered_clusters,
            ordered_edges,
            distance_matrix: None,
            adjacency_matrix: None,
            components: None,
            frontier_sizes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn compute_distance_matrix(&self) -> Vec<Vec<U>> {
        let indices: HashMap<_, _> = self.ordered_clusters.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut matrix: Vec<Vec<U>> = vec![vec![U::zero(); self.vertex_cardinality()]; self.vertex_cardinality()];
        self.ordered_edges.iter().for_each(|&e| {
            let i = *indices.get(e.left()).unwrap();
            let j = *indices.get(e.right()).unwrap();
            matrix[i][j] = e.distance();
            matrix[j][i] = e.distance();
        });
        matrix
    }

    pub fn with_distance_matrix(mut self) -> Self {
        self.distance_matrix = Some(self.compute_distance_matrix());
        self
    }

    pub fn with_adjacency_matrix(mut self) -> Self {
        self.adjacency_matrix = Some(
            self.distance_matrix()
                .iter()
                .map(|row| row.iter().map(|v| !v.is_zero()).collect())
                .collect(),
        );
        self
    }

    fn filtered_edges(&self, clusters: &ClusterSet<T, U>) -> EdgeSet<T, U> {
        self.ordered_edges
            .iter()
            .filter(|e| clusters.contains(e.left()) && clusters.contains(e.right()))
            .copied()
            .collect()
    }

    pub fn with_components(&'a mut self) -> Self {
        let mut components = Vec::new();

        let mut unvisited: Vec<_> = self.ordered_clusters.to_vec();
        while !unvisited.is_empty() {
            let start = unvisited.pop().unwrap();
            let (visited, _) = self.traverse(start);
            unvisited = unvisited.into_iter().filter(|c| !visited.contains(c)).collect();
            let edges = self.filtered_edges(&visited);
            components.push(Graph::new(visited, edges));
        }

        Graph {
            space: self.space,
            cluster_set: self.cluster_set.clone(),
            edge_set: self.edge_set.clone(),
            adjacency_map: self.adjacency_map.clone(),
            population: self.population,
            min_depth: self.min_depth,
            max_depth: self.max_depth,
            ordered_clusters: self.ordered_clusters.clone(),
            ordered_edges: self.ordered_edges.clone(),
            distance_matrix: self.distance_matrix.clone(),
            adjacency_matrix: self.adjacency_matrix.clone(),
            components: Some(components),
            frontier_sizes: self.frontier_sizes.clone(),
        }
    }

    pub fn space(&self) -> &dyn Space<T, U> {
        self.space
    }

    pub fn clusters(&self) -> &[&Cluster<T, U>] {
        &self.ordered_clusters
    }

    pub fn cluster_set(&self) -> &ClusterSet<T, U> {
        &self.cluster_set
    }

    pub fn edges(&self) -> &[&Edge<T, U>] {
        &self.ordered_edges
    }

    pub fn edge_set(&self) -> &EdgeSet<T, U> {
        &self.edge_set
    }

    pub fn vertex_cardinality(&self) -> usize {
        self.cluster_set.len()
    }

    pub fn edge_cardinality(&self) -> usize {
        self.edges().len()
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

    pub fn adjacency_map(&'a self) -> &AdjacencyMap<T, U> {
        &self.adjacency_map
    }

    pub fn diameter(&'a self) -> usize {
        self.cluster_set
            .iter()
            .map(|&c| self.eccentricity(c).unwrap())
            .max()
            .unwrap()
    }

    pub fn distance_matrix(&self) -> &[Vec<U>] {
        self.distance_matrix
            .as_ref()
            .expect("Please call `with_distance_matrix` on the Graph before using `distance_matrix`.")
    }

    pub fn adjacency_matrix(&self) -> &[Vec<bool>] {
        self.adjacency_matrix
            .as_ref()
            .expect("Please call `with_adjacency_matrix` on the Graph before using `adjacency_matrix`.")
    }

    pub fn components(&self) -> &Vec<Graph<'a, T, U>> {
        self.components
            .as_ref()
            .expect("Please call `with_distance_matrix` on the Graph before using `distance_matrix`.")
    }

    fn assert_contains(&self, c: &Cluster<T, U>) -> Result<(), String> {
        if self.cluster_set.contains(c) {
            Ok(())
        } else {
            Err(format!("Cluster {:} is not in this graph.", c))
        }
    }

    pub fn vertex_degree(&self, c: &Cluster<T, U>) -> Result<usize, String> {
        Ok(self.neighbors_of(c)?.len())
    }

    pub fn neighbors_of(&self, c: &Cluster<T, U>) -> Result<Vec<&Cluster<T, U>>, String> {
        self.assert_contains(c)?;
        Ok(self.adjacency_map.get(c).unwrap().iter().copied().collect())
    }

    fn traverse(&self, start: &'a Cluster<T, U>) -> (HashSet<&'a Cluster<T, U>>, Vec<usize>) {
        let mut visited: HashSet<&Cluster<T, U>> = HashSet::new();
        let mut frontier: HashSet<&Cluster<T, U>> = HashSet::new();
        frontier.insert(start);
        let mut frontier_sizes: Vec<usize> = Vec::new();

        while !frontier.is_empty() {
            visited.extend(frontier.iter().copied());
            frontier = frontier
                .iter()
                .flat_map(|&c| self.neighbors_of(c).unwrap())
                .filter(|&n| !((visited.contains(n)) || (frontier.contains(n))))
                .collect();
            frontier_sizes.push(frontier.len());
        }

        (visited, frontier_sizes)
    }

    pub fn frontier_sizes(&'a self, c: &'a Cluster<T, U>) -> Result<Vec<usize>, String> {
        self.assert_contains(c)?;
        let n = self
            .frontier_sizes
            .read()
            .unwrap()
            .get(c)
            .cloned()
            .unwrap_or_else(|| self.traverse(c).1);
        self.frontier_sizes.write().unwrap().insert(c, n.clone());
        Ok(n)
    }

    pub fn eccentricity(&'a self, c: &'a Cluster<T, U>) -> Result<usize, String> {
        Ok(self.frontier_sizes(c)?.len())
    }
}
