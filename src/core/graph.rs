use std::collections::HashMap;
use std::collections::HashSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::RwLock;

use dashmap::DashMap;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;

type ClusterSet<T, U> = HashSet<Arc<Cluster<T, U>>>;
type ClusterVec<T, U> = Vec<Arc<Cluster<T, U>>>;
type EdgeSet<T, U> = HashSet<Arc<Edge<T, U>>>;
type Components<T, U> = Arc<RwLock<Option<Vec<Arc<Graph<T, U>>>>>>;

type EdgesMap<T, U> = HashMap<Arc<Cluster<T, U>>, EdgeSet<T, U>>;

/// A HashMap where the value is a HashSet of clusters that are subsumed by the key cluster.
pub type Subsumed<T, U> = HashMap<Arc<Cluster<T, U>>, ClusterSet<T, U>>;

/// Two clusters have an edge between them if the distance between their centers is less than or equal to the sum of their radii.
#[derive(Debug)]
pub struct Edge<T: Number, U: Number> {
    pub left: Arc<Cluster<T, U>>,
    pub right: Arc<Cluster<T, U>>,
    pub distance: U,
}

/// Two edges are the same if they connect the same two clusters.
impl<T: Number, U: Number> PartialEq for Edge<T, U> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

impl<T: Number, U: Number> Eq for Edge<T, U> {}

/// This may be used for creating dot files for graphviz, or for hashing an edge.
impl<T: Number, U: Number> std::fmt::Display for Edge<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:} -- {:}, {:}", self.left.name, self.right.name, self.distance)
    }
}

impl<T: Number, U: Number> Hash for Edge<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:}", self).hash(state)
    }
}

impl<T: Number, U: Number> Edge<T, U> {
    /// Create a new edge between two clusters. It is the user's responsibility to check the distance between the cluster centers.
    pub fn new(left: Arc<Cluster<T, U>>, right: Arc<Cluster<T, U>>, distance: U) -> Arc<Self> {
        let edge = if format!("{}", left) < format!("{}", right) {
            Edge { left, right, distance }
        } else {
            Edge { right, left, distance }
        };
        Arc::new(edge)
    }

    /// Returns whether the edge is from a cluster to itself.
    pub fn to_self(&self) -> bool {
        (self.distance == U::zero()) || (self.left == self.right)
    }

    /// Returns whether the given cluster is one of the end points of the edge.
    pub fn contains(&self, cluster: &Arc<Cluster<T, U>>) -> bool {
        cluster == &self.left || cluster == &self.right
    }

    /// Given one of the clusters in the edge, returns the other cluster.
    /// Returns an Err if the given cluster is not contained in the edge.
    pub fn neighbor(&self, cluster: &Arc<Cluster<T, U>>) -> Result<&Arc<Cluster<T, U>>, String> {
        if cluster == &self.left {
            Ok(&self.right)
        } else if cluster == &self.right {
            Ok(&self.left)
        } else {
            let message = format!("Cluster {:} is not in this edge.", cluster.name);
            Err(message)
        }
    }
}

/// A Graph can be induced from a collection of clusters by having a one-to-one correspondence with vertices, and adding an edge between any two clusters whose volumes overlap with each other.
/// 
/// TODO: Implement Display in dot-file format.
#[derive(Debug)]
pub struct Graph<T: Number, U: Number> {
    // TODO: Measure performance difference for using a Vec instead of a HashSet.
    // TODO: Improve performance of hashing and lookup of the expensive properties of the graph, e.g. edges-dict, components, eccentricities, etc.
    
    /// A HashSet of clusters/vertices in the graph.
    pub clusters: ClusterSet<T, U>,

    /// A HashSet of edges in the graph.
    pub edges: EdgeSet<T, U>,

    /// The number of clusters/vertices in the graph.
    pub cardinality: usize,

    /// The sum of cardinalities of all clusters/vertices in the graph.
    pub population: usize,

    /// The maximum tree-depth of any cluster/vertex in the graph.
    pub depth: usize,

    /// The minimum tree-depth of any cluster/vertex in the graph.
    pub min_depth: usize,

    /// A HashMap from a cluster to all edges of that cluster.
    pub edges_dict: EdgesMap<T, U>,

    /// A collection of all connected components of the graph. Each component is itself a graph.
    components: Components<T, U>,

    /// A HashMap of clusters and their eccentricities.
    eccentricities: DashMap<Arc<Cluster<T, U>>, usize>,

    /// Name of the distance metric used in the dataset.
    pub metric_name: String,
}
// TODO: Implement Display, perhaps using Dot-String format

/// Two graphs are the same if they have the same set of clusters.
impl<T: Number, U: Number> PartialEq for Graph<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}

impl<T: Number, U: Number> Eq for Graph<T, U> {}

impl<T: Number, U: Number> Graph<T, U> {
    /// Creates a new graph.
    /// 
    /// TODO: Lazily compute the graph properties.
    pub fn new(clusters: HashSet<Arc<Cluster<T, U>>>, edges: EdgeSet<T, U>) -> Self {
        assert!(!clusters.is_empty(), "Must have at least one cluster to make a graph.");
        let metric_name = clusters.iter().next().unwrap().clone().dataset.metric().name();
        let mut graph = Graph {
            clusters,
            edges,
            cardinality: 0,
            population: 0,
            depth: 0,
            min_depth: 0,
            edges_dict: HashMap::new(),
            components: Arc::new(RwLock::new(None)),
            eccentricities: DashMap::new(),
            metric_name,
        };
        graph.cardinality = graph.cardinality();
        graph.population = graph.population();
        graph.depth = graph.depth();
        graph.min_depth = graph.min_depth();
        graph.edges_dict = graph.edges_dict();
        graph
    }

    fn cardinality(&self) -> usize {
        self.clusters.len()
    }

    fn population(&self) -> usize {
        self.clusters.par_iter().map(|cluster| cluster.cardinality).sum()
    }

    fn depth(&self) -> usize {
        self.clusters.par_iter().map(|cluster| cluster.depth()).max().unwrap()
    }

    fn min_depth(&self) -> usize {
        self.clusters.par_iter().map(|cluster| cluster.depth()).min().unwrap()
    }

    /// Returns the minimum and maximum tree-depths of the clusters in the graph.
    pub fn depth_range(&self) -> (usize, usize) {
        (self.min_depth, self.depth)
    }

    fn edges_dict(&self) -> EdgesMap<T, U> {
        self.clusters
            .par_iter()
            .map(|cluster| {
                (
                    Arc::clone(cluster),
                    self.edges
                        .par_iter()
                        .filter(|&edge| edge.contains(cluster))
                        .map(Arc::clone)
                        .collect(),
                )
            })
            .collect()
    }

    fn assert_contains(&self, cluster: &Arc<Cluster<T, U>>) -> Result<(), String> {
        if self.clusters.contains(cluster) {
            Ok(())
        } else {
            Err(format!("This Graph does not contain the Cluster {}.", cluster.name))
        }
    }

    /// Returns a HashSet of edges from the given cluster.
    pub fn edges_from(&self, cluster: &Arc<Cluster<T, U>>) -> Result<&EdgeSet<T, U>, String> {
        self.assert_contains(cluster)?;
        Ok(self.edges_dict.get(cluster).unwrap())
    }

    /// Returns the neighbors of the given cluster.
    pub fn neighbors(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        Ok((self.edges_from(cluster)?)
            .par_iter()
            .map(|edge| edge.neighbor(cluster).unwrap())
            .map(Arc::clone)
            .collect())
    }

    /// Returns the distances to each neighbor of the given cluster.
    pub fn distances(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<U>, String> {
        Ok((self.edges_from(cluster)?)
            .par_iter()
            .map(|edge| edge.distance)
            .collect())
    }

    /// Returns the subgraph of the given subset of clusters.
    pub fn subgraph(&self, cluster_subset: HashSet<Arc<Cluster<T, U>>>) -> Result<Self, String> {
        for cluster in cluster_subset.iter() {
            self.assert_contains(cluster)?;
        }

        let edges = self
            .edges
            .par_iter()
            .filter(|&edge| cluster_subset.contains(&edge.left) && cluster_subset.contains(&edge.right))
            .map(Arc::clone)
            .collect();

        Ok(Graph::new(cluster_subset, edges))
    }

    /// Perform a graph traversal (in an arbitrary order) starting at the given cluster.
    /// 
    /// Returns the set of visited clusters and the eccentricity of the strating cluster.
    fn traverse(&self, start: &Arc<Cluster<T, U>>) -> (ClusterSet<T, U>, usize) {
        let mut visited = HashSet::new();

        let mut frontier = HashSet::new();
        frontier.insert(Arc::clone(start));

        let mut eccentricity = 0;

        while !frontier.is_empty() {
            let new_frontier = frontier
                .par_iter()
                .map(|c| {
                    self.neighbors(c)
                        .unwrap()
                        .par_iter()
                        .filter(|&n| !visited.contains(n) && !frontier.contains(n))
                        .map(Arc::clone)
                        .collect::<Vec<Arc<Cluster<T, U>>>>()
                })
                .flatten()
                .collect();

            visited.extend(frontier);
            frontier = new_frontier;
            eccentricity += 1;
        }

        (visited, eccentricity)
    }

    /// Returns the connected components of the graph.
    pub fn find_components(&self) -> Vec<Arc<Self>> {
        let components: Option<Vec<Arc<Self>>> = self.components.read().unwrap().clone();
        if components.is_none() {
            let mut components = Vec::new();
            let mut unvisited = self.clusters.clone();

            while !unvisited.is_empty() {
                let start = Arc::clone(unvisited.iter().next().unwrap());
                let (component, _) = self.traverse(&start);
                component
                    .iter()
                    .map(|cluster_item| unvisited.remove(cluster_item))
                    .count();
                components.push(self.subgraph(component).unwrap());
            }

            let components = components.into_iter().map(Arc::new).collect();
            *self.components.write().unwrap() = Some(components);
        }

        self.components.read().unwrap().clone().unwrap()
    }

    /// Returns the eccentricity of a cluster, i.e. the maximum length of all paths starting at the cluster.
    pub fn eccentricity(&self, cluster: &Arc<Cluster<T, U>>) -> Result<usize, String> {
        self.assert_contains(cluster)?;

        if !self.eccentricities.contains_key(cluster) {
            let (_, eccentricity) = self.traverse(cluster);
            self.eccentricities.insert(Arc::clone(cluster), eccentricity);
        }

        Ok(*self.eccentricities.get(cluster).unwrap().value())
    }

    /// Returns the graph diameter, i.e. the maximum eccentricity of any cluster in the graph.
    pub fn diameter(&self) -> usize {
        self.clusters
            .par_iter()
            .map(|cluster_item| self.eccentricity(cluster_item).unwrap())
            .max()
            .unwrap()
    }

    /// Returns a vec of the clusters in the graph and a matrix of pairwise distances between the clsuter centers.
    /// The rows and columns of the matrix are ordered by the vec of clusters.
    pub fn distance_matrix(&self) -> (ClusterVec<T, U>, Array2<U>) {
        let clusters: Vec<_> = self.clusters.par_iter().map(Arc::clone).collect();
        let indices: HashMap<_, _> = clusters
            .par_iter()
            .map(Arc::clone)
            .enumerate()
            .map(|(i, cluster)| (cluster, i))
            .collect();
        let mut matrix = Array2::zeros((self.cardinality, self.cardinality));
        for edge in self.edges.iter() {
            let (&i, &j) = (indices.get(&edge.left).unwrap(), indices.get(&edge.right).unwrap());
            matrix[[i, j]] = edge.distance;
            matrix[[j, i]] = edge.distance;
        }
        (clusters, matrix)
    }

    /// Returns the adhjacency matrix for the graph, along with a vec of clsuters in the same order.
    pub fn adjacency_matrix(&self) -> (ClusterVec<T, U>, Array2<bool>) {
        let (clusters, distances) = self.distance_matrix();
        (clusters, distances.mapv(|v| v > U::zero()))
    }

    /// Returns a pruned subgraph, i.e. after removing all subsumed clusters, along with HashMap noting all subsumed clusters.
    /// 
    /// A cluster is said to be subsumed by another cluster if its volume liex completely inside the others volume.
    pub fn pruned_graph(&self) -> (Arc<Self>, Subsumed<T, U>) {
        let subsumed_clusters: HashSet<_> = self
            .edges
            .par_iter()
            .flat_map(|edge| {
                if edge.distance + edge.left.radius < edge.right.radius {
                    vec![Arc::clone(&edge.left)]
                } else if edge.distance + edge.right.radius < edge.left.radius {
                    vec![Arc::clone(&edge.right)]
                } else {
                    vec![]
                }
            })
            .collect();

        let pruned_clusters: HashSet<_> = self
            .clusters
            .par_iter()
            .filter(|&cluster| !subsumed_clusters.contains(cluster))
            .map(Arc::clone)
            .collect();

        let subsumed_neighbors = pruned_clusters
            .par_iter()
            .map(|cluster| {
                let neighbors = self
                    .neighbors(cluster)
                    .unwrap()
                    .into_par_iter()
                    .filter(|neighbor| !pruned_clusters.contains(neighbor))
                    .collect();
                (Arc::clone(cluster), neighbors)
            })
            .collect();

        let pruned_graph = Arc::new(self.subgraph(pruned_clusters).unwrap());

        (pruned_graph, subsumed_neighbors)
    }

    /// Returns the connected component containing the given cluster.
    pub fn component_containing(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Arc<Self>, String> {
        self.assert_contains(cluster)?;
        Ok(self
            .find_components()
            .into_par_iter()
            .find_any(|component| component.clusters.contains(cluster))
            .unwrap())
    }
}
