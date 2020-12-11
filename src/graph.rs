use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dashmap::{DashMap, DashSet};

use crate::cluster::Cluster;
use crate::dataset::Dataset;
use crate::metric::Real;
use crate::types::*;

pub type EdgesDict<T, U> = DashMap<Arc<Cluster<T, U>>, Arc<DashSet<Arc<Edge<T, U>>>>>;

#[derive(Debug)]
pub struct Edge<T: Real, U: Real> {
    pub left: Arc<Cluster<T, U>>,
    pub right: Arc<Cluster<T, U>>,
    pub distance: U,
}

impl<T: Real, U: Real> PartialEq for Edge<T, U> {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

impl<T: Real, U: Real> Eq for Edge<T, U> {}

impl<T: Real, U: Real> fmt::Display for Edge<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:} -- {:}, {:}", self.left.name, self.right.name, self.distance)
    }
}

impl<T: Real, U: Real> Hash for Edge<T, U> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:}", self).hash(state)
    }
}

impl<T: Real, U: Real> Edge<T, U> {
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


pub struct Graph<T: Real, U: Real> {
    pub dataset: Arc<Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
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

impl<T: Real, U: Real> PartialEq for Graph<T, U> {
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

impl<T: Real, U: Real> Eq for Graph<T, U> {}

// impl fmt::Display for Graph {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         let display = self.clusters.iter().map(|c| format!("{:}", c)).collect::<Vec<String>>().join("");
//         write!(f, "{:}", display)
//     }
// }

impl<T: Real, U: Real> Graph<T, U> {
    pub fn new(root: Arc<Cluster<T, U>>, clusters: DashSet<Arc<Cluster<T, U>>>) -> Result<Graph<T, U>, String> {
        assert!(!clusters.is_empty(), "Must have at least one cluster to make a graph.");
        let mut graph = Graph {
            dataset: Arc::clone(&root.dataset),
            root,
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
        graph.build_edges()?;
        graph.edges_dict = Arc::new(graph.edges_dict());
        Ok(graph)
    }

    fn cardinality(&self) -> Index {
        self.clusters.len()
    }

    fn population(&self) -> Index {
        self.clusters.iter().map(|cluster| cluster.cardinality()).sum()
    }

    fn indices(&self) -> Indices {
        self.clusters.iter().map(|cluster| cluster.indices.clone()).flatten().collect()
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

    fn depth(&self) -> usize {
        self.clusters.iter().map(|cluster| cluster.depth()).max().unwrap()
    }

    fn min_depth(&self) -> usize {
        self.clusters.iter().map(|cluster| cluster.depth()).min().unwrap()
    }

    pub fn depth_range(&self) -> (usize, usize) {
        (self.min_depth(), self.depth())
    }

    fn build_edges(&self) -> Result<(), String> {
        for item in self.clusters.iter() {
            self._find_neighbors(item.key())?;
        }
        Ok(())
    }

    fn _find_candidates(&self, cluster: &Arc<Cluster<T, U>>) -> Result<(), String> {
        let mut radius = self.root.radius;
        let mut grand_ancestor = Arc::clone(&self.root);

        for depth in 1..(cluster.depth() + 1) {
            let ancestor = grand_ancestor.descend_towards(cluster.name.borrow())?;
            if ancestor.radius > U::zero() {
                radius = ancestor.radius
            }

            if ancestor.candidates.len() == 0 {
                let ancestor_candidates = Arc::clone(&grand_ancestor.candidates);
                let potential_candidates: Arc<DashSet<Arc<Cluster<T, U>>>> = Arc::new(
                    ancestor_candidates
                        .iter()
                        .map(|item| Arc::clone(item.key()))
                        .collect()
                );
                ancestor_candidates
                    .iter()
                    .for_each(|item| if item.key().depth() == depth - 1 {
                        potential_candidates.insert(Arc::clone(item.key()));
                    });

                potential_candidates
                    .iter()
                    .for_each(|item| {
                        let distance = ancestor.distance_to(item.key());
                        if distance <= item.key().radius + radius * U::from_f64(4.).unwrap() {
                            ancestor.candidates.insert(Arc::clone(item.key()), distance);
                        }
                    });
            }

            grand_ancestor = ancestor
        }
        Ok(())
    }

    fn _find_neighbors(&self, cluster: &Arc<Cluster<T, U>>) -> Result<(), String> {
        if cluster.candidates.len() == 0 {
            self._find_candidates(cluster)?;
        }

        for item in cluster.candidates.iter() {
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
}
