use std::sync::Arc;

use crate::cluster::Cluster;
use crate::criteria::ClusterCriterion;
use crate::dataset::Dataset;
use crate::graph::Graph;

pub struct Manifold {
    pub dataset: Arc<Dataset>,
    pub root: Arc<Cluster>,
    pub layers: Vec<Arc<Graph>>,
}

impl Manifold {
    pub fn new(dataset: Arc<Dataset>, cluster_criteria: Vec<impl ClusterCriterion>) -> Manifold {
        let cluster_criteria = cluster_criteria
            .into_iter()
            .map(Arc::new)
            .collect();
        let indices = dataset.indices();
        let root = Cluster::new(
            Arc::clone(&dataset),
            "".to_string(),
            indices,
        ).partition(&cluster_criteria);
        Manifold {
            dataset,
            root: Arc::new(root),
            layers: vec![],
        }
    }
}
