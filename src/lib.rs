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
    use crate::utils::read_test_data;

    #[test]
    fn test_cluster() {
        let data: Array2<f64> = array![[1., 2., 3.], [3., 3., 1.]];
        let dataset = Arc::new(Dataset::<f64, f64>::new(data, "euclidean", false).unwrap());
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

    #[test]
    fn test_search() {
        let data: Array2<f64> = arr2(&[
            [0., 0.],
            [1., 1.],
            [2., 2.],
            [3., 3.],
        ]);
        let dataset = Arc::new(Dataset::new(data, "euclidean", false).unwrap());
        let search = Search::build(Arc::clone(&dataset), None);

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
        let (data, _) = read_test_data();
        let dataset = Arc::new(Dataset::<f64, f64>::new(data, "euclidean", false).unwrap());
        let cluster = Cluster::new(
            Arc::clone(&dataset),
            "".to_string(),
            dataset.indices(),
            // increase depth for longer benchmark
        ).partition(&vec![criteria::MaxDepth::new(6)]);
        assert!(cluster.num_descendents() > 50);
    }
}
