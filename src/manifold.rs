use std::sync::Arc;

use crate::cluster::Cluster;
use crate::dataset::Dataset;
use crate::graph::Graph;
use crate::metric::Number;
use crate::criteria::ClusterCriterion;

pub struct Manifold<T: Number, U: Number> {
    pub dataset: Arc<Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
    pub layers: Vec<Arc<Graph<T, U>>>,
    pub graphs: Vec<Arc<Graph<T, U>>>,
}

impl<T: Number, U: Number> Manifold<T, U> {
    pub fn new(dataset: Arc<Dataset<T, U>>, cluster_criteria: Vec<Box<impl ClusterCriterion>>) -> Manifold<T, U> {
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
            graphs: vec![],
        }
    }

    pub fn ancestry(&self, name: &String) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        let mut ancestry = vec![Arc::clone(&self.root)];
        if !name.is_empty() {
            for depth in 1..(name.len() + 1) {
                let child = ancestry[depth - 1].descend_towards(&name[0..depth].to_string())?;
                ancestry.push(child);
            }
        }
        Ok(ancestry)
    }

    pub fn select(&self, name: &String) -> Result<Arc<Cluster<T, U>>, String> {
        let ancestry = self.ancestry(name)?;
        Ok(Arc::clone(&ancestry[name.len()]))
    }
}
