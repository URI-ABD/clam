use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::cluster::Cluster;
use crate::dataset::Dataset;
use crate::metric::Real;
use crate::types::*;

pub type EdgesDict<T, U> = HashMap<Arc<Cluster<T, U>>, Vec<Arc<Edge<T, U>>>>;

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

    #[allow(clippy::type_complexity)]
    pub fn clusters(&self) -> (&Arc<Cluster<T, U>>, &Arc<Cluster<T, U>>) {
        (&self.left, &self.right)
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
    pub clusters: Vec<Arc<Cluster<T, U>>>,
    pub edges: Vec<Arc<Edge<T, U>>>,
    pub is_built: bool,
    pub cardinality: Index,
    pub population: Index,
    pub indices: Indices,
    pub depth: usize,
    pub min_depth: usize,
    pub edges_dict: EdgesDict<T, U>,
}

impl<T: Real, U: Real> PartialEq for Graph<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.clusters == other.clusters
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
    pub fn new(clusters: Vec<Arc<Cluster<T, U>>>) -> Graph<T, U> {
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
        let mut edges_dict = HashMap::new();
        for cluster in self.clusters.iter() {
            edges_dict.insert(
                Arc::clone(cluster),
                self.edges
                .iter()
                .filter(|&edge| edge.contains(cluster))
                .cloned()
                .collect::<Vec<Arc<Edge<T, U>>>>()
            );
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

    // fn find_candidates(
    //     &self,
    //     cluster: &Arc<Cluster>,
    //     candidates_map: HashMap<Arc<Cluster>, Vec<(Arc<Cluster>, f64)>>,
    // ) {
    //
    // }
}
