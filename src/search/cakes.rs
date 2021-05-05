use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;
use std::{fmt, result};

use ndarray::prelude::*;
use rayon::prelude::*;

use crate::prelude::*;

/// A Vec of Clusters that overlap with the query ball.
pub type ClusterHits<T, U> = Vec<Arc<Cluster<T, U>>>;

/// A HashMap of indices of all hits and their distances to the query.
pub type Hits<T> = HashMap<Index, T>;

/// CLAM-Augmented K-nearest-neighbors Entropy-scaling Search
///
/// Provides tools for similarity search.
/// Search time scales sub-linearly with the size of the dataset.
/// This is orders of magnitude faster than state-of-the-art tools while also
/// guaranteeing exact results (as compared to naive linear search)
/// when the distance function used obeys the triangle inequality.
///
/// Paper pending...
///
/// TODO: Add Compression and Decompression for the dataset and search tree.
///
/// TODO: Add Serde support for storing and loading the search tree.
pub struct Cakes<T: Number, U: Number> {
    /// An Arc to any struct that implements the `Dataset` trait.
    pub dataset: Arc<dyn Dataset<T, U>>,

    /// The root Cluster of the search tree.
    root: Arc<Cluster<T, U>>,

    /// The distance function being used.
    distance: Metric<T, U>,
}

impl<T: Number, U: Number> fmt::Debug for Cakes<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Search").field("dataset", &self.dataset).finish()
    }
}

impl<T: Number, U: Number> Cakes<T, U> {
    /// Builds a search tree for the given dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - An Arc to any struct that implements the `Dataset` trait.
    /// * `max_depth` - Clusters in the tree that have a higher depth will not be partitioned.
    ///                 Capped at 63 until I feel like bothering with bit-vectors.
    /// * `min_cardinality` - Clusters in the tree that have a smaller cardinality will not be partitioned.
    pub fn build(
        dataset: Arc<dyn Dataset<T, U>>,
        max_depth: Option<u8>,
        min_cardinality: Option<usize>,
    ) -> Cakes<T, U> {
        // parse the max-depth and min-cardinality and create the partition-criterion.
        let criteria = vec![
            criteria::max_depth(std::cmp::min(max_depth.unwrap_or(63), 63)),
            criteria::min_cardinality(min_cardinality.unwrap_or(1)),
        ];
        // build the search tree.
        let root = Cluster::new(Arc::clone(&dataset), 1, dataset.indices()).partition(&criteria);
        // return the struct
        Cakes {
            dataset: Arc::clone(&dataset),
            root: Arc::new(root),
            distance: metric_new(dataset.metric()).unwrap(),
        }
    }

    /// Returns the diameter of the search tree, a useful property for judging appropriate search radii.
    pub fn diameter(&self) -> U {
        U::from(2).unwrap() * self.root.radius
    }

    /// Performs accelerated rho-nearest search on the dataset and
    /// returns all hits inside a sphere of the given `radius` centered at the requested `query`.
    pub fn rnn(&self, query: Arc<ArrayView<T, IxDyn>>, radius: Option<U>) -> Hits<U> {
        self.leaf_search(Arc::clone(&query), radius, self.tree_search(Arc::clone(&query), radius))
    }

    /// Performs coarse-grained tree-search to find all clusters that could potentially contain hits.
    pub fn tree_search(&self, query: Arc<ArrayView<T, IxDyn>>, radius: Option<U>) -> ClusterHits<T, U> {
        // parse the search radius
        let radius = radius.unwrap_or_else(U::zero);
        // if query ball has overlapping volume with the root, delegate to the recursive, private method.
        if (self.distance)(self.root.center(), Arc::clone(&query)) <= (radius + self.root.radius) {
            self._tree_search(&self.root, query, radius)
        } else {
            // otherwise, return an empty Vec signifying no possible hits.
            vec![]
        }
    }

    //noinspection DuplicatedCode
    fn _tree_search(
        &self,
        cluster: &Arc<Cluster<T, U>>,
        query: Arc<ArrayView<T, IxDyn>>,
        radius: U,
    ) -> ClusterHits<T, U> {
        // Invariant: Entering this function means that the current cluster has overlapping volume with the query-ball.
        // Invariant: Triangle-inequality guarantees exactness of results from each recursive call.
        match cluster.children.borrow() {
            // There are children. Make recursive calls if necessary.
            Some((left, right)) => {
                // get the two vectors of hits from up to two recursive calls.
                let (mut l, mut r) = rayon::join(
                    || {
                        // If the child has overlap with the query-ball, recurse into the child
                        if self.query_distance(Arc::clone(&query), left.argcenter) <= (radius + left.radius) {
                            self._tree_search(&left, Arc::clone(&query), radius)
                        } else {
                            // otherwise return an empty vec.
                            vec![]
                        }
                    },
                    || {
                        if self.query_distance(Arc::clone(&query), right.argcenter) <= (radius + right.radius) {
                            self._tree_search(&right, Arc::clone(&query), radius)
                        } else {
                            vec![]
                        }
                    },
                );
                // combine both Vectors into one.
                l.append(&mut r);
                l
            }
            None => {
                // There are no children so return a Vec containing only the current cluster.
                vec![Arc::clone(cluster)]
            }
        }
    }

    /// Exhaustively searches the clusters identified by tree-search and
    /// returns a HashMap of all hits and their distance from the query.
    pub fn leaf_search(
        &self,
        query: Arc<ArrayView<T, IxDyn>>,
        radius: Option<U>,
        clusters: ClusterHits<T, U>,
    ) -> Hits<U> {
        let indices = clusters
            .iter()
            .map(|c| c.indices.clone())
            .into_iter()
            .flatten()
            .collect::<Indices>();
        self.linear_search(query, radius, Some(indices))
    }

    /// Naive search. Useful for leaf-search and for measuring acceleration from entropy-scaling search.
    pub fn linear_search(
        &self,
        query: Arc<ArrayView<T, IxDyn>>,
        radius: Option<U>,
        indices: Option<Indices>,
    ) -> Hits<U> {
        let radius = radius.unwrap_or_else(U::zero);
        let indices = indices.unwrap_or_else(|| self.dataset.indices());
        indices
            .par_iter()
            .map(|&i| (i, self.query_distance(Arc::clone(&query), i)))
            .filter(|(_, d)| *d <= radius)
            .collect()
    }

    // A convenient wrapper to get the distance from a given query to an indexed instance in the dataset.
    fn query_distance(&self, query: Arc<ArrayView<T, IxDyn>>, index: Index) -> U {
        (self.distance)(query, self.dataset.instance(index))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::prelude::*;

    use crate::dataset::RowMajor;
    use crate::prelude::*;
    use crate::utils::read_test_data;

    use super::Cakes;

    #[test]
    fn test_search() {
        let data: Array2<f64> = arr2(&[[0., 0.], [1., 1.], [2., 2.], [3., 3.]]);
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::new(data, "euclidean", false).unwrap());
        let search = Cakes::build(Arc::clone(&dataset), None, None);

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
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::<f64, f64>::new(data, "euclidean", true).unwrap());

        let search = Cakes::build(Arc::clone(&dataset), Some(50), None);

        let search_str = [
            "Search { ".to_string(),
            format!("dataset: {:?}", dataset),
            " }".to_string(),
        ]
        .join("");
        assert_eq!(format!("{:?}", search), search_str);

        let radius = Some(search.diameter() / 100.);

        for &q in dataset.indices()[0..10].iter() {
            let query = Arc::new(dataset.instance(q));

            let cakes_results = search.rnn(Arc::clone(&query), radius.clone());
            let naive_results = search.linear_search(Arc::clone(&query), radius.clone(), None);

            let no_extra = cakes_results.iter().all(|(i, _)| naive_results.contains_key(i));
            assert!(
                no_extra,
                "had some extras {} / {}",
                naive_results.len(),
                cakes_results.len()
            );

            let no_misses = naive_results.iter().all(|(i, _)| cakes_results.contains_key(i));
            assert!(
                no_misses,
                "had some misses {} / {}",
                naive_results.len(),
                cakes_results.len()
            );
        }
    }
}
