use std::{fmt, result};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::ArrayView1;

use crate::cluster::Cluster;
use crate::criteria;
use crate::dataset::Dataset;
use crate::metric::Metric;
use crate::types::*;

type ClusterResults = HashMap<Arc<Cluster>, f64>;
type Results = HashMap<Index, f64>;

pub struct Search {
    pub dataset: Arc<Dataset>,
    root: Arc<Cluster>,
    distance: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
}

impl fmt::Debug for Search {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Search")
            .field("dataset", &self.dataset)
            .finish()
    }
}

impl Search {
    pub fn new(dataset: Arc<Dataset>, max_depth: Option<usize>) -> Search {
        let _criteria = match max_depth {
            Some(d) => vec![criteria::MaxDepth::new(d)],
            None => vec![],
        };
        let indices = dataset.indices();
        let root = Cluster::new(Arc::clone(&dataset), "".to_string(), indices)
            .partition(&_criteria);
        let metric = &(*dataset.metric);
        Search {
            dataset,
            root: Arc::new(root),
            distance: Metric::on_f64(metric),
        }
    }

    pub fn indices(&self) -> Indices { self.dataset.indices() }

    pub fn rnn(&self, query: ArrayView1<f64>, radius: Option<f64>) -> Results {
        self.leaf_search(query, radius, self.tree_search(query, radius))
    }

    pub fn tree_search(&self, query: ArrayView1<f64>, radius: Option<f64>) -> ClusterResults {
        let radius = radius.unwrap_or(0.);
        let mut results: ClusterResults = HashMap::new();
        self._tree_search(&self.root, query, radius, &mut results);
        results
    }

    fn _tree_search(&self, cluster: &Arc<Cluster>, query: ArrayView1<f64>, radius: f64, results: &mut ClusterResults) {
        let distance = self.query_distance(query, cluster.center());
        if distance <= radius + cluster.radius() {
            match cluster.children.borrow() {
                Some(children) => {
                    for child in children.iter() {
                        self._tree_search(child, query, radius, results);
                    }
                },
                None => {
                    results.insert(Arc::clone(cluster), distance);
                }
            }
        }
    }

    pub fn leaf_search(&self, query: ArrayView1<f64>, radius: Option<f64>, clusters: ClusterResults) -> Results {
        let radius = radius.unwrap_or(0.);
        let mut results: Results = HashMap::new();
        for (cluster, _) in clusters.iter() {
            for &i in cluster.indices.iter() {
                let point = self.dataset.row(i);
                let distance = self.query_distance(point, query);
                if distance <= radius { results.insert(i, distance); }
            }
        }
        results
    }

    fn query_distance(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> f64 {
        (self.distance)(left, right)
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ndarray::{arr2, Array2};

    use crate::dataset::Dataset;

    use super::Search;

    #[test]
    fn test_search() {
        let data: Array2<f64> = arr2(&[
            [0., 0.],
            [1., 1.],
            [2., 2.],
            [3., 3.],
        ]);
        let dataset = Dataset::new(data, "euclidean");
        let search_str = ["Search { ".to_string(), format!("dataset: {:?}", dataset), " }".to_string()].join("");
        let search = Search::new(Arc::new(dataset), Some(10));

        assert_eq!(search.indices().len(), 4);
        assert_eq!(format!("{:?}", search), search_str);
    }
}
