use std::sync::Arc;

use crate::cluster::Cluster;
use crate::criteria::ClusterCriterion;
use crate::dataset::Dataset;
use crate::graph::Graph;
use crate::metric::Real;

pub struct Manifold<T: Real, U: Real> {
    pub dataset: Arc<Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
    pub layers: Vec<Arc<Graph<T, U>>>,
}

impl<T: Real, U: Real> Manifold<T, U> {
    pub fn new(dataset: Arc<Dataset<T, U>>, cluster_criteria: Vec<impl ClusterCriterion>) -> Manifold<T, U> {
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
