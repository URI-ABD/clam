use std::sync::Arc;

use crate::cluster::Cluster;
use crate::criteria::ClusterCriterion;
use crate::dataset::Dataset;
use crate::graph::Graph;
use crate::metric::Number;

pub struct Manifold<T: Number, U: Number> {
    pub dataset: Arc<Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
    pub layers: Vec<Arc<Graph<T, U>>>,
}

impl<T: Number, U: Number> Manifold<T, U> {
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

    pub fn ancestry(&self, name: &str) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        let mut ancestry = vec![Arc::clone(&self.root)];
        if !name.is_empty() {
            for depth in 1..(name.len() + 1) {
                let child = ancestry[depth - 1].descend_towards(&name[0..depth])?;
                ancestry.push(child);
            }
        }
        Ok(ancestry)
    }

    pub fn select(&self, name: &str) -> Result<Arc<Cluster<T, U>>, String> {
        let ancestry = self.ancestry(name)?;
        Ok(Arc::clone(&ancestry[name.len()]))
    }
}
