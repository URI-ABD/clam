use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dashmap::{DashMap, DashSet};
use ndarray::Array2;

use crate::cluster::Cluster;
use crate::metric::Number;
use crate::types::{CandidatesMap, EdgesDict, Index, Indices};

type Subsumed<T, U> = DashMap<Arc<Cluster<T, U>>, HashSet<Cluster<T, U>>>;

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

impl<T: Number, U: Number> fmt::Display for Edge<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:} -- {:}, {:}", self.left.name, self.right.name, self.distance)
    }
}

impl<T: Number, U: Number> Hash for Edge<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:}", self).hash(state)
    }
}

impl<T: Number, U: Number> Edge<T, U> {
    pub fn new(left: Arc<Cluster<T, U>>, right: Arc<Cluster<T, U>>, distance: U) -> Edge<T, U> {
        Edge {
            left,
            right,
            distance,
        }
    }

    pub fn to_self(&self) -> bool {
        self.left == self.right
    }

    pub fn contains(&self, cluster: &Arc<Cluster<T, U>>) -> bool {
        cluster == &self.left || cluster == &self.right
    }

    pub fn neighbor(&self, cluster: &Arc<Cluster<T, U>>) -> Result<&Arc<Cluster<T, U>>, String> {
        if cluster == &self.left { Ok(&self.right) }
        else if cluster == &self.right { Ok(&self.left) }
        else {
            let message = format!("Cluster {:} is not in this edge.", cluster.name);
            Err(message)
        }
    }
}


#[derive(Debug)]
pub struct Graph<T: Number, U: Number> {
    pub clusters: DashSet<Arc<Cluster<T, U>>>,
    pub edges: Arc<DashSet<Arc<Edge<T, U>>>>,
    pub is_built: bool,
    pub cardinality: Index,
    pub population: Index,
    pub indices: Indices,
    pub depth: usize,
    pub min_depth: usize,
    pub edges_dict: Arc<EdgesDict<T, U>>,
}
// TODO: Implement Display, perhaps using Dot-String format

impl<T: Number, U: Number> PartialEq for Graph<T, U> {
    fn eq(&self, other: &Self) -> bool {
        let left = self.clusters
            .iter()
            .filter(|item| other.clusters.contains(item.key()))
            .count();
        let right = other.clusters
            .iter()
            .filter(|item| self.clusters.contains(item.key()))
            .count();
        left == right && right == 0
    }
}

impl<T: Number, U: Number> Eq for Graph<T, U> {}

impl<T: Number, U: Number> Graph<T, U> {
    pub fn new(clusters: DashSet<Arc<Cluster<T, U>>>) -> Result<Graph<T, U>, String> {
        assert!(!clusters.is_empty(), "Must have at least one cluster to make a graph.");
        let mut graph = Graph {
            clusters,
            edges: Arc::new(DashSet::new()),
            is_built: false,
            cardinality: 0,
            population: 0,
            indices: vec![],
            depth: 0,
            min_depth: 0,
            edges_dict: Arc::new(DashMap::new()),
        };
        graph.cardinality = graph.cardinality();
        graph.population = graph.population();
        graph.indices = graph.indices();
        graph.depth = graph.depth();
        graph.min_depth = graph.min_depth();
        Ok(graph)
    }

    fn cardinality(&self) -> Index {
        self.clusters.len()
    }

    fn population(&self) -> Index {
        self.clusters.iter().map(|cluster| cluster.cardinality()).sum()
    }

    fn indices(&self) -> Indices {
        self.clusters
            .iter()
            .map(|cluster| cluster.indices.clone())
            .flatten()
            .collect()
    }

    fn depth(&self) -> usize {
        self.clusters
            .iter()
            .map(|cluster| cluster.depth())
            .max()
            .unwrap()
    }

    fn min_depth(&self) -> usize {
        self.clusters
            .iter()
            .map(|cluster| cluster.depth())
            .min()
            .unwrap()
    }

    pub fn depth_range(&self) -> (usize, usize) {
        (self.min_depth(), self.depth())
    }

    fn _find_candidates(&self, root: &Arc<Cluster<T, U>>, cluster: &Arc<Cluster<T, U>>, candidates_map: &Arc<CandidatesMap<T, U>>) -> Result<(), String> {
        let mut radius = root.radius;
        let mut grand_ancestor = Arc::clone(root);

        for depth in 1..(cluster.depth() + 1) {
            let ancestor = grand_ancestor.descend_towards(cluster.name.borrow())?;
            radius = if ancestor.radius > U::zero() { ancestor.radius } else { radius };

            if !candidates_map.contains_key(&ancestor) {
                let potential_candidates = DashSet::new();

                let grand_ancestor_candidates = candidates_map.get(&grand_ancestor).unwrap();
                grand_ancestor_candidates
                    .iter()
                    .for_each(|candidate| {
                        potential_candidates.insert(Arc::clone(candidate.key()));
                    });
                grand_ancestor_candidates
                    .iter()
                    .for_each(|candidate| if candidate.key().depth() == depth - 1 {
                        match candidate.key().children.borrow() {
                            Some((left, right)) => {
                                potential_candidates.insert(Arc::clone(&left));
                                potential_candidates.insert(Arc::clone(&right));
                            }
                            None => ()
                        };
                    });

                let ancestor_candidates = DashMap::new();
                potential_candidates
                    .iter()
                    .for_each(|candidate| {
                        let distance = ancestor.distance_to(candidate.key());
                        if distance <= candidate.key().radius + radius * U::from(4).unwrap() {
                            ancestor_candidates.insert(Arc::clone(candidate.key()), distance);
                        }
                    });
                candidates_map.insert(Arc::clone(&ancestor), ancestor_candidates);
            }
            grand_ancestor = ancestor
        }
        Ok(())
    }

    fn _find_neighbors(&self, root: &Arc<Cluster<T, U>>, cluster: &Arc<Cluster<T, U>>, candidates_map: &Arc<CandidatesMap<T, U>>) -> Result<(), String> {
        if !candidates_map.contains_key(cluster) {
            self._find_candidates(root, cluster, candidates_map)?;
        }
        let candidates = candidates_map.get(cluster).unwrap();

        for item in candidates.iter() {
            let (candidate, &distance) = (item.key(), item.value());
            if (cluster != candidate)
                && (self.clusters.contains(candidate))
                && (distance <= cluster.radius + candidate.radius) {
                let edge = Edge::new(Arc::clone(cluster), Arc::clone(candidate), distance);
                    self.edges.insert(Arc::new(edge));
            }
        }
        Ok(())
    }

    fn edges_dict(&self) -> EdgesDict<T, U> {
        let edges_dict = DashMap::new();
        for cluster_item in self.clusters.iter() {
            let edges = DashSet::new();
            self.edges
                .iter()
                .for_each(|edge_item| if edge_item.key().contains(cluster_item.key()) {
                    edges.insert(Arc::clone(edge_item.key()));
                });
            edges_dict.insert(Arc::clone(cluster_item.key()), Arc::new(edges));
        }
        edges_dict
    }

    pub fn build_edges(&mut self, root: &Arc<Cluster<T, U>>, candidates_map: &Arc<CandidatesMap<T, U>>) -> Result<(), String> {
        for item in self.clusters.iter() {
            self._find_neighbors(root, item.key(), candidates_map)?;
        }
        self.edges_dict = Arc::new(self.edges_dict());
        self.is_built = true;
        Ok(())
    }

    fn assert_contains(&self, cluster: &Arc<Cluster<T, U>>) -> Result<(), String> {
        if self.clusters.contains(cluster) {
            Ok(())
        } else {
            Err(format!("Cluster {} is not in this graph.", cluster.name))
        }
    }

    fn assert_built(&self) -> Result<(), String> {
        if self.is_built {
            Ok(())
        } else {
            Err("Edges have not yet been built for this graph.".to_string())
        }
    }

    pub fn edges_from(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Arc<DashSet<Arc<Edge<T, U>>>>, String> {
        self.assert_contains(cluster)?;
        self.assert_built()?;
        Ok(Arc::clone(self.edges_dict.get(cluster).unwrap().borrow()))
    }

    pub fn neighbors(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        let mut neighbors = Vec::new();
        (self.edges_from(cluster)?)
            .iter()
            .for_each(|edge_item| {
                neighbors.push(Arc::clone(edge_item.key().neighbor(cluster).unwrap()));
            });
        Ok(neighbors)
    }

    pub fn distances(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<U>, String> {
        let mut distances = Vec::new();
        (self.edges_from(cluster)?)
            .iter()
            .for_each(|edge_item| {
                distances.push(edge_item.key().distance);
            });
        Ok(distances)
    }

    pub fn subgraph(&self, clusters: DashSet<Arc<Cluster<T, U>>>) -> Result<Graph<T, U>, String> {
        for cluster in clusters.iter() {
            self.assert_contains(cluster.key())?;
        }
        self.assert_built()?;

        let edges = DashSet::new();
        self.edges.iter().for_each(|edge_item| {
            let edge = edge_item.key();
            if clusters.contains(&edge.left) && clusters.contains(&edge.right) {
                edges.insert(Arc::clone(edge));
            }
        });

        let mut graph = Graph::new(clusters)?;
        graph.edges = Arc::new(edges);
        graph.edges_dict = Arc::new(graph.edges_dict());
        graph.is_built = true;
        Ok(graph)
    }

    fn traverse(&self, start: &Arc<Cluster<T, U>>) -> Result<(DashSet<Arc<Cluster<T, U>>>, usize), String> {
        self.assert_contains(start)?;
        self.assert_built()?;

        let mut visited = DashSet::new();
        let mut frontier = DashSet::new();
        frontier.insert(Arc::clone(start));
        let mut eccentricity = 0;

        loop {
            let new_frontier = DashSet::new();
            for cluster in frontier.iter() {
                let neighbors = self.neighbors(cluster.key())?;
                for neighbor in neighbors.iter() {
                    if !visited.contains(neighbor) && !frontier.contains(neighbor) {
                        new_frontier.insert(Arc::clone(neighbor));
                    }
                }
            }

            if new_frontier.is_empty() {
                break
            } else {
                eccentricity += 1;
                visited.extend(frontier);
                frontier = new_frontier
            }
        }
        Ok((visited, eccentricity))
    }

    pub fn components(&self) -> Result<Vec<Graph<T, U>>, String> {
        self.assert_built()?;

        let mut components = Vec::new();
        let unvisited = self.clusters.clone();

        loop {
            let start = Arc::clone(unvisited.iter().next().unwrap().key());
            let (component, _) = self.traverse(&start).unwrap();
            component
                .iter()
                .for_each(|cluster_item| {
                    unvisited.remove(cluster_item.key());
                });
            components.push(self.subgraph(component)?);
            if unvisited.is_empty() {
                break
            }
        }

        Ok(components)
    }

    pub fn eccentricity(&self, cluster: &Arc<Cluster<T, U>>) -> Result<usize, String> {
        let (_, eccentricity) = self.traverse(cluster)?;
        Ok(eccentricity)
    }

    pub fn diameter(&self) -> usize {
        self.clusters
            .iter()
            .map(|cluster_item| self.eccentricity(cluster_item.key()).unwrap())
            .max()
            .unwrap()
    }

    pub fn distance_matrix(&self) -> Result<(Vec<Arc<Cluster<T, U>>>, Array2<U>), String> {
        self.assert_built()?;

        let clusters: Vec<Arc<Cluster<T, U>>> = self.clusters
            .iter()
            .map(|cluster_item| Arc::clone(cluster_item.key()))
            .collect();
        let mut indices = HashMap::new();
        clusters
            .iter()
            .enumerate()
            .for_each(|(i, cluster)| {
                indices.insert(Arc::clone(cluster), i).unwrap();
            });
        let mut matrix = Array2::zeros((clusters.len(), clusters.len()));
        for edge_item in self.edges.iter() {
            let edge = edge_item.key();
            let (&i, &j) = (indices.get(&edge.left).unwrap(), indices.get(&edge.right).unwrap());
            matrix[[i, j]] = edge.distance;
            matrix[[j, i]] = edge.distance;
        }
        Ok((clusters, matrix))
    }

    pub fn adjacency_matrix(&self) -> Result<(Vec<Arc<Cluster<T, U>>>, Array2<bool>), String> {
        let (clusters, distances) = self.distance_matrix()?;
        Ok((clusters, distances.mapv(|v| v > U::zero())))
    }

    pub fn pruned_graph(&self) -> Result<(Graph<T, U>, Subsumed<T, U>), String> {
        self.assert_built()?;
        unimplemented!()
    }

    pub fn component_containing(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Graph<T, U>, String> {
        self.assert_contains(cluster)?;
        self.assert_built()?;
        unimplemented!()
    }
}
