pub mod types;
pub mod utils;
pub mod metric;
pub mod dataset;
pub mod criteria;
pub mod cluster;
pub mod graph;
pub mod manifold;
pub mod search;

// TODO: Use ndarray_npy crate to read large arrays for more testing

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::prelude::*;

    use crate::cluster::Cluster;
    use crate::criteria;
    use crate::dataset::Dataset;
    use crate::search::Search;
    use crate::utils::{DATASETS, read_data};

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 3., 1.]];
        let dataset = Dataset::new(data, "euclidean");
        let indices = dataset.indices();
        let cluster = Cluster::new(Arc::new(dataset), "".to_string(), indices)
            .partition(&vec![criteria::MaxDepth::new(3)]);

        assert_eq!(cluster.depth(), 0);
        assert_eq!(cluster.cardinality(), 2);
        assert_eq!(cluster.num_descendents(), 2);
        assert!(cluster.radius > 0.);
        assert!(cluster.contains(&0));
        assert!(cluster.contains(&1));

        let children = cluster.children.unwrap();
        for child in children.iter() {
            assert_eq!(child.depth(), 1);
            assert_eq!(child.cardinality(), 1);
            assert_eq!(child.num_descendents(), 0);
        }
    }

    #[test]
    fn test_search() {
        let data: Array2<f64> = arr2(&[
            [0., 0.],
            [1., 1.],
            [2., 2.],
            [3., 3.],
        ]);
        let dataset = Dataset::new(data, "euclidean");
        let search = Search::new(Arc::new(dataset), Some(10));

        let q = arr1(&[0., 1.]);
        let query: ArrayView1<f64> = q.view();
        let results = search.rnn(query, Some(1.5));
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(!results.contains_key(&2));
        assert!(!results.contains_key(&3));

        let query = search.dataset.row(1);
        let results = search.rnn(query, None);
        assert_eq!(results.len(), 1);
        assert!(!results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(!results.contains_key(&2));
        assert!(!results.contains_key(&3));
    }

    #[test]
    fn test_large_array() {
        let dataset = DATASETS[0];
        let (data, _) = read_data(dataset).unwrap();
        let dataset = Dataset::new(data, "euclidean");
        let indices = dataset.indices();
        let cluster = Cluster::new(
            Arc::new(dataset),
            "".to_string(),
            indices,
        ).partition(&vec![criteria::MaxDepth::new(6)]);
        println!("{:}", cluster.num_descendents());
        assert!(cluster.num_descendents() > 50);
    }
}
