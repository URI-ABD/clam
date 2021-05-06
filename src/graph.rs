use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;

pub type Candidates<T, U> = HashMap<Arc<Cluster<T, U>>, U>;
pub type CandidatesMap<T, U> = HashMap<Arc<Cluster<T, U>>, Candidates<T, U>>;

pub type ClusterSet<T, U> = HashSet<Arc<Cluster<T, U>>>;
pub type EdgeSet<T, U> = HashSet<Arc<Edge<T, U>>>;

pub type EdgesMap<T, U> = HashMap<Arc<Cluster<T, U>>, EdgeSet<T, U>>;
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
        Edge { left, right, distance }
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
    pub clusters: ClusterSet<T, U>,
    pub edges: EdgeSet<T, U>,
    pub is_built: bool,
    pub cardinality: Index,
    pub population: Index,
    pub indices: Indices,
    pub depth: u8,
    pub min_depth: u8,
    pub edges_dict: EdgesMap<T, U>,
}
// TODO: Implement Display, perhaps using Dot-String format

impl<T: Number, U: Number> PartialEq for Graph<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}

impl<T: Number, U: Number> Eq for Graph<T, U> {}

impl<T: Number, U: Number> Graph<T, U> {
    pub fn new(clusters: HashSet<Arc<Cluster<T, U>>>) -> Result<Graph<T, U>, String> {
        assert!(!clusters.is_empty(), "Must have at least one cluster to make a graph.");
        let mut graph = Graph {
            clusters,
            edges: HashSet::new(),
            is_built: false,
            cardinality: 0,
            population: 0,
            indices: vec![],
            depth: 0,
            min_depth: 0,
            edges_dict: HashMap::new(),
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
        self.clusters.iter().map(|cluster| cluster.cardinality).sum()
    }

    fn indices(&self) -> Indices {
        self.clusters
            .par_iter()
            .map(|cluster| cluster.indices.clone())
            .flatten()
            .collect()
    }

    fn depth(&self) -> u8 {
        self.clusters.par_iter().map(|cluster| cluster.depth()).max().unwrap()
    }

    fn min_depth(&self) -> u8 {
        self.clusters.par_iter().map(|cluster| cluster.depth()).min().unwrap()
    }

    pub fn depth_range(&self) -> (u8, u8) {
        (self.min_depth, self.depth)
    }

    pub fn find_candidates(
        &self,
        _root: &Arc<Cluster<T, U>>,
        _cluster: &Arc<Cluster<T, U>>,
        _candidates_map: &Arc<CandidatesMap<T, U>>,
    ) -> Result<(), String> {
        // TODO: Think about doing this in Manifold
        unimplemented!()
        // let mut radius = root.radius;
        // let mut grand_ancestor = Arc::clone(root);

        // for depth in 1..(cluster.depth() + 1) {
        //     let ancestor = grand_ancestor.descend_towards(cluster.name.borrow())?;
        //     radius = if ancestor.radius > U::zero() {
        //         ancestor.radius
        //     } else {
        //         radius
        //     };

        //     if !candidates_map.contains_key(&ancestor) {
        //         let potential_candidates = DashSet::new();

        //         let grand_ancestor_candidates = candidates_map.get(&grand_ancestor).unwrap();
        //         grand_ancestor_candidates.iter().for_each(|candidate| {
        //             potential_candidates.insert(Arc::clone(candidate.key()));
        //         });
        //         grand_ancestor_candidates.iter().for_each(|candidate| {
        //             if candidate.key().depth() == depth - 1 {
        //                 match candidate.key().children.borrow() {
        //                     Some((left, right)) => {
        //                         potential_candidates.insert(Arc::clone(&left));
        //                         potential_candidates.insert(Arc::clone(&right));
        //                     }
        //                     None => (),
        //                 };
        //             }
        //         });

        //         let ancestor_candidates = DashMap::new();
        //         potential_candidates.iter().for_each(|candidate| {
        //             let distance = ancestor.distance_to(candidate.key());
        //             if distance <= candidate.key().radius + radius * U::from(4).unwrap() {
        //                 ancestor_candidates.insert(Arc::clone(candidate.key()), distance);
        //             }
        //         });
        //         candidates_map.insert(Arc::clone(&ancestor), ancestor_candidates);
        //     }
        //     grand_ancestor = ancestor
        // }
        // Ok(())
    }

    pub fn find_neighbors(
        &self,
        _root: &Arc<Cluster<T, U>>,
        _cluster: &Arc<Cluster<T, U>>,
        _candidates_map: &Arc<CandidatesMap<T, U>>,
    ) -> Result<(), String> {
        unimplemented!()
        // if !candidates_map.contains_key(cluster) {
        //     self.find_candidates(root, cluster, candidates_map)?;
        // }
        // let candidates = candidates_map.get(cluster).unwrap();

        // for item in candidates.iter() {
        //     let (candidate, &distance) = (item.key(), item.value());
        //     if (cluster != candidate)
        //         && (self.clusters.contains(candidate))
        //         && (distance <= cluster.radius + candidate.radius)
        //     {
        //         let edge = Edge::new(Arc::clone(cluster), Arc::clone(candidate), distance);
        //         self.edges.insert(Arc::new(edge));
        //     }
        // }
        // Ok(())
    }

    fn edges_dict(&self) -> EdgesMap<T, U> {
        self.clusters
            .par_iter()
            .map(|c| {
                (
                    Arc::clone(c),
                    self.edges
                        .iter()
                        .filter(|&e| e.contains(c))
                        .map(|e| Arc::clone(e))
                        .collect(),
                )
            })
            .collect()
    }

    pub fn build_edges(&mut self) -> Result<(), String> {
        unimplemented!()
        // for item in self.clusters.iter() {
        //     self.find_neighbors(root, item.key(), candidates_map)?;
        // }
        // self.edges_dict = Arc::new(self.edges_dict());
        // self.is_built = true;
        // Ok(())
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

    #[allow(clippy::type_complexity)]
    pub fn edges_from(&self, cluster: &Arc<Cluster<T, U>>) -> Result<&EdgeSet<T, U>, String> {
        self.assert_contains(cluster)?;
        self.assert_built()?;
        Ok(self.edges_dict.get(cluster).unwrap())
    }

    pub fn neighbors(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        let mut neighbors = Vec::new();
        (self.edges_from(cluster)?).iter().for_each(|edge_item| {
            neighbors.push(Arc::clone(edge_item.neighbor(cluster).unwrap()));
        });
        Ok(neighbors)
    }

    pub fn distances(&self, cluster: &Arc<Cluster<T, U>>) -> Result<Vec<U>, String> {
        let mut distances = Vec::new();
        (self.edges_from(cluster)?).iter().for_each(|edge_item| {
            distances.push(edge_item.distance);
        });
        Ok(distances)
    }

    pub fn subgraph(&self, cluster_set: HashSet<Arc<Cluster<T, U>>>) -> Result<Graph<T, U>, String> {
        for cluster in cluster_set.iter() {
            self.assert_contains(cluster)?;
        }
        self.assert_built()?;

        let edges = self
            .edges
            .iter()
            .filter(|&e| cluster_set.contains(&e.left) && cluster_set.contains(&e.right))
            .cloned()
            .collect();

        let mut graph = Graph::new(cluster_set)?;
        graph.edges = edges;
        graph.edges_dict = graph.edges_dict();
        graph.is_built = true;
        Ok(graph)
    }

    #[allow(clippy::type_complexity)]
    fn traverse(&self, start: &Arc<Cluster<T, U>>) -> Result<(ClusterSet<T, U>, usize), String> {
        self.assert_contains(start)?;
        self.assert_built()?;

        let mut visited: ClusterSet<T, U> = HashSet::new();
        let mut frontier: ClusterSet<T, U> = HashSet::new();
        frontier.insert(Arc::clone(start));
        let mut eccentricity = 0;

        loop {
            let new_frontier: ClusterSet<T, U> = frontier
                .iter()
                .map(|c| {
                    self.neighbors(c)
                        .unwrap()
                        .iter()
                        .filter(|&n| !visited.contains(n) && !frontier.contains(n))
                        .cloned()
                        .collect::<Vec<Arc<Cluster<T, U>>>>()
                })
                .flatten()
                .collect();

            if new_frontier.is_empty() {
                break;
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
        let mut unvisited = self.clusters.clone();

        loop {
            let start = Arc::clone(unvisited.iter().next().unwrap());
            let (component, _) = self.traverse(&start).unwrap();
            component.iter().for_each(|cluster_item| {
                unvisited.remove(cluster_item);
            });
            components.push(self.subgraph(component)?);
            if unvisited.is_empty() {
                break;
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
            .map(|cluster_item| self.eccentricity(cluster_item).unwrap())
            .max()
            .unwrap()
    }

    #[allow(clippy::type_complexity)]
    pub fn distance_matrix(&self) -> Result<(Vec<Arc<Cluster<T, U>>>, Array2<U>), String> {
        self.assert_built()?;

        let clusters: Vec<Arc<Cluster<T, U>>> = self
            .clusters
            .iter()
            .map(|cluster_item| Arc::clone(cluster_item))
            .collect();
        let mut indices = HashMap::new();
        clusters.iter().enumerate().for_each(|(i, cluster)| {
            indices.insert(Arc::clone(cluster), i).unwrap();
        });
        let mut matrix = Array2::zeros((clusters.len(), clusters.len()));
        for edge_item in self.edges.iter() {
            let edge = edge_item;
            let (&i, &j) = (indices.get(&edge.left).unwrap(), indices.get(&edge.right).unwrap());
            matrix[[i, j]] = edge.distance;
            matrix[[j, i]] = edge.distance;
        }
        Ok((clusters, matrix))
    }

    #[allow(clippy::type_complexity)]
    pub fn adjacency_matrix(&self) -> Result<(Vec<Arc<Cluster<T, U>>>, Array2<bool>), String> {
        let (clusters, distances) = self.distance_matrix()?;
        Ok((clusters, distances.mapv(|v| v > U::zero())))
    }

    #[allow(clippy::type_complexity)]
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
