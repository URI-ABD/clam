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
pub type Subsumed<T, U> = HashMap<Arc<Cluster<T, U>>, ClusterSet<T, U>>;

#[derive(Debug)]
pub struct Edge<T: Number, U: Number> {
    pub left: Arc<Cluster<T, U>>,
    pub right: Arc<Cluster<T, U>>,
    pub distance: U,
}

impl<T: Number, U: Number> PartialEq for Edge<T, U> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

impl<T: Number, U: Number> Eq for Edge<T, U> {}

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
    pub fn new(left: Arc<Cluster<T, U>>, right: Arc<Cluster<T, U>>, distance: U) -> Arc<Self> {
        let edge = if format!("{}", left) < format!("{}", right) {
            Edge { left, right, distance }
        } else {
            Edge { right, left, distance }
        };
        Arc::new(edge)
    }

    pub fn to_self(&self) -> bool {
        self.left == self.right
    }

    pub fn contains(&self, cluster: &Arc<Cluster<T, U>>) -> bool {
        cluster == &self.left || cluster == &self.right
    }

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

#[derive(Debug)]
pub struct Graph<T: Number, U: Number> {
    // TODO: Measure difference for using a Vec instead of a HashSet
    pub clusters: ClusterSet<T, U>,
    pub edges: EdgeSet<T, U>,
    pub cardinality: usize,
    pub population: usize,
    pub depth: usize,
    pub min_depth: usize,
    pub edges_dict: EdgesMap<T, U>,
    components: Components<T, U>,
    eccentricities: DashMap<Arc<Cluster<T, U>>, usize>,
    pub metric_name: String,
}
// TODO: Implement Display, perhaps using Dot-String format

impl<T: Number, U: Number> PartialEq for Graph<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}

impl<T: Number, U: Number> Eq for Graph<T, U> {}

impl<T: Number, U: Number> Graph<T, U> {
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

    pub fn edges_from(&self, cluster: &Arc<Cluster<T, U>>) -> Result<&EdgeSet<T, U>, String> {
        self.assert_contains(cluster)?;
        Ok(self.edges_dict.get(cluster).unwrap())
    }

    pub fn neighbors(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        Ok((self.edges_from(cluster)?)
            .par_iter()
            .map(|edge| edge.neighbor(cluster).unwrap())
            .map(Arc::clone)
            .collect())
    }

    pub fn distances(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<U>, String> {
        Ok((self.edges_from(cluster)?)
            .par_iter()
            .map(|edge| edge.distance)
            .collect())
    }

    pub fn subgraph(&self, cluster_set: HashSet<Arc<Cluster<T, U>>>) -> Result<Self, String> {
        for cluster in cluster_set.iter() {
            self.assert_contains(cluster)?;
        }

        let edges = self
            .edges
            .par_iter()
            .filter(|&edge| cluster_set.contains(&edge.left) && cluster_set.contains(&edge.right))
            .map(Arc::clone)
            .collect();

        Ok(Graph::new(cluster_set, edges))
    }

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

    pub fn eccentricity(&self, cluster: &Arc<Cluster<T, U>>) -> Result<usize, String> {
        self.assert_contains(cluster)?;

        if !self.eccentricities.contains_key(cluster) {
            let (_, eccentricity) = self.traverse(cluster);
            self.eccentricities.insert(Arc::clone(cluster), eccentricity);
        }

        Ok(*self.eccentricities.get(cluster).unwrap().value())
    }

    pub fn diameter(&self) -> usize {
        self.clusters
            .par_iter()
            .map(|cluster_item| self.eccentricity(cluster_item).unwrap())
            .max()
            .unwrap()
    }

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

    pub fn adjacency_matrix(&self) -> (ClusterVec<T, U>, Array2<bool>) {
        let (clusters, distances) = self.distance_matrix();
        (clusters, distances.mapv(|v| v > U::zero()))
    }

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

    pub fn component_containing(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Arc<Self>, String> {
        self.assert_contains(cluster)?;
        Ok(self
            .find_components()
            .into_par_iter()
            .find_any(|component| component.clusters.contains(cluster))
            .unwrap())
    }
}
