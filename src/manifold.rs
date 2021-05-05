use std::sync::Arc;

use crate::prelude::*;
use criteria::PartitionCriterion;

pub struct Manifold<T: Number, U: Number> {
    pub dataset: Arc<dyn Dataset<T, U>>,
    pub root: Arc<Cluster<T, U>>,
    pub layers: Vec<Arc<Graph<T, U>>>,
    pub graphs: Vec<Arc<Graph<T, U>>>,
}

impl<T: Number, U: Number> Manifold<T, U> {
    pub fn new(dataset: Arc<dyn Dataset<T, U>>, cluster_criteria: Vec<PartitionCriterion<T, U>>) -> Manifold<T, U> {
        let indices = dataset.indices();
        let root = Cluster::new(Arc::clone(&dataset), 1, indices).partition(&cluster_criteria);
        Manifold {
            dataset,
            root: Arc::new(root),
            layers: vec![],
            graphs: vec![],
        }
    }

    pub fn ancestry(&self, _name: u64) -> Result<Vec<Arc<Cluster<T, U>>>, String> {
        unimplemented!()
        // let mut ancestry = vec![Arc::clone(&self.root)];
        // if !name.is_empty() {
        //     for depth in 1..(name.len() + 1) {
        //         let child = ancestry[depth - 1].descend_towards(&name[0..depth].to_string())?;
        //         ancestry.push(child);
        //     }
        // }
        // Ok(ancestry)
    }

    pub fn select(&self, _name: u64) -> Result<Arc<Cluster<T, U>>, String> {
        unimplemented!()
        // let ancestry = self.ancestry(name)?;
        // Ok(Arc::clone(&ancestry[name as usize]))
    }
}
