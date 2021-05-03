pub use cluster::Cluster;
pub use graph::{Edge, Graph};
pub use manifold::Manifold;

pub mod criteria;
mod cluster;
mod graph;
mod manifold;



#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::prelude::*;

    use crate::dataset::RowMajor;
    use crate::prelude::*;

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 3., 1.]];
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::<f64, f64>::new(
            data,
            "euclidean",
            false,
        ).unwrap());
        let indices = dataset.indices();
        let cluster = Cluster::new(Arc::clone(&dataset), "".to_string(), indices)
            .partition(&vec![criteria::MaxDepth::new(3)]);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 2);
        assert_eq!(cluster.num_descendents(), 2);
        assert!(cluster.radius > 0.);
        assert!(cluster.contains(&0));
        assert!(cluster.contains(&1));

        let (left, right) = cluster.children.unwrap();
        for child in [left, right].iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 1);
            assert_eq!(child.num_descendents(), 0);
        }
    }
}
