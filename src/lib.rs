pub mod metric;
pub mod dataset;
pub mod clam;
pub mod cakes;
pub mod chaoda;
pub mod prelude;
pub mod utils;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::prelude::*;

    use crate::prelude::*;
    use crate::dataset::RowMajor;
    use crate::cakes::Search;
    use crate::utils::read_test_data;

    #[test]
    fn test_search() {
        let data: Array2<f64> = arr2(&[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]);
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::new(
            data,
            "euclidean",
            false,
        ).unwrap());
        let search = Search::build(Arc::clone(&dataset), None);

        let q = arr1(&[0., 1.]);
        let query: Arc<ArrayView<f64, IxDyn>> = Arc::new(q.view().into_dyn());
        let results = search.rnn(Arc::clone(&query), Some(1.5));
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(!results.contains_key(&2));
        assert!(!results.contains_key(&3));

        let query = Arc::new(search.dataset.instance(1));
        let results = search.rnn(Arc::clone(&query), None);
        assert_eq!(results.len(), 1);
        assert!(!results.contains_key(&0));
        assert!(results.contains_key(&1));
        assert!(!results.contains_key(&2));
        assert!(!results.contains_key(&3));
    }

    #[test]
    fn test_search_large() {
        let (data, _) = read_test_data();
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::<f64, f64>::new(
            data,
            "euclidean",
            false,
        ).unwrap());
        let cluster = Cluster::new(
            Arc::clone(&dataset),
            "".to_string(),
            dataset.indices(),
            // increase depth for longer benchmark
        )
        .partition(&vec![criteria::MaxDepth::new(6)]);
        assert!(cluster.num_descendents() > 50);
    }
}
