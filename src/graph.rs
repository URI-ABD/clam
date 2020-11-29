use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::cluster::Cluster;
use crate::types::*;
use std::collections::HashMap;
use crate::dataset::Dataset;

#[derive(Debug)]
pub struct Edge {
    pub left: Arc<Cluster>,
    pub right: Arc<Cluster>,
    pub distance: f64,
}

impl PartialEq for Edge {
    fn eq(&self, other: &Self) -> bool {
        (self.left == other.left) && (self.right == other.right)
    }
}

impl Eq for Edge {}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:} -- {:}, {:}", self.left.name, self.right.name, self.distance)
    }
}

impl Hash for Edge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:}", self).hash(state)
    }
}

impl Edge {
    pub fn new(left: Arc<Cluster>, right: Arc<Cluster>, distance: f64) -> Edge {
        Edge {
            left,
            right,
            distance,
        }
    }

    pub fn to_self(&self) -> bool {
        self.left == self.right
    }

    pub fn contains(&self, cluster: &Arc<Cluster>) -> bool {
        cluster == &self.left || cluster == &self.right
    }

    pub fn clusters(&self) -> (&Arc<Cluster>, &Arc<Cluster>) {
        (&self.left, &self.right)
    }

    pub fn neighbor(&self, cluster: &Arc<Cluster>) -> Result<&Arc<Cluster>, String> {
        if cluster == &self.left { Ok(&self.right) }
        else if cluster == &self.right { Ok(&self.left) }
        else {
            let message = format!("Cluster {:} is not in this edge.", cluster.name);
            Err(message)
        }
    }
}


pub struct Graph {
    pub dataset: Arc<Dataset>,
    pub clusters: Vec<Arc<Cluster>>,
    pub edges: Vec<Arc<Edge>>,
    pub is_built: bool,
    pub cardinality: usize,
    pub population: usize,
    pub indices: Indices,
    pub depth: usize,
    pub min_depth: usize,
    pub edges_dict: HashMap<Arc<Cluster>, Vec<Arc<Edge>>>,
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
    }
}

impl Eq for Graph {}

// impl fmt::Display for Graph {
//     fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
//         let display = self.clusters.iter().map(|c| format!("{:}", c)).collect::<Vec<String>>().join("");
//         write!(f, "{:}", display)
//     }
// }

impl Graph {
    pub fn new(clusters: Vec<Arc<Cluster>>) -> Graph {
        assert!(!clusters.is_empty(), "Must have at least one cluster to make a graph.");
        let mut graph = Graph {
            dataset: Arc::clone(&clusters[0].dataset),
            clusters,
            edges: vec![],
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
        graph.edges_dict = graph.edges_dict();
        graph
    }

    fn cardinality(&self) -> usize { self.clusters.len() }

    fn population(&self) -> usize { self.clusters.iter().map(|cluster| cluster.cardinality()).sum() }

    fn indices(&self) -> Indices { self.clusters.iter().map(|cluster| cluster.indices.clone()).flatten().collect() }

    fn edges_dict(&self) -> HashMap<Arc<Cluster>, Vec<Arc<Edge>>> {
        let mut edges_dict = HashMap::new();
        for cluster in self.clusters.iter() {
            edges_dict.insert(
                Arc::clone(cluster),
                self.edges
                .iter()
                .filter(|&edge| edge.contains(cluster))
                .cloned()
                .collect::<Vec<Arc<Edge>>>()
            );
        }
        edges_dict
    }

    fn depth(&self) -> usize { self.clusters.iter().map(|cluster| cluster.depth()).max().unwrap() }

    fn min_depth(&self) -> usize { self.clusters.iter().map(|cluster| cluster.depth()).min().unwrap() }

    pub fn depth_range(&self) -> (usize, usize) { (self.min_depth(), self.depth()) }

    // fn find_candidates(
    //     &self,
    //     cluster: &Arc<Cluster>,
    //     candidates_map: HashMap<Arc<Cluster>, Vec<(Arc<Cluster>, f64)>>,
    // ) {
    //
    // }
}
