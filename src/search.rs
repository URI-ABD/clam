use std::{fmt, result};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::cluster::Cluster;
use crate::criteria;
use crate::dataset::Dataset;
use crate::metric::Metric;
use crate::types::{Index, Indices};

type ClusterResults = HashMap<Arc<Cluster>, f64>;
type Results = DashMap<Index, f64>;

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
        let criteria = match max_depth {
            Some(d) => vec![criteria::MaxDepth::new(d)],
            None => vec![],
        };
        let indices = dataset.indices();
        let root = Cluster::new(
            Arc::clone(&dataset),
            "".to_string(),
            indices,
        ).partition(&criteria);
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
        self._tree_search(&self.root, query, radius)
    }

    fn _tree_search(&self, cluster: &Arc<Cluster>, query: ArrayView1<f64>, radius: f64) -> ClusterResults {
        let distance = self.query_distance(query, cluster.center());
        let mut results: ClusterResults = HashMap::new();
        if distance <= radius + cluster.radius {
            let hits = match cluster.children.borrow() {
                Some((left, right)) => {
                    let (mut m1, m2) = rayon::join(
                        || self._tree_search(left, query, radius),
                        || self._tree_search(right, query, radius),
                    );
                    m1.extend(m2);
                    m1
                },
                None => {
                    let mut m: ClusterResults = HashMap::new();
                    m.insert(Arc::clone(cluster), distance);
                    m
                }
            };
            results.extend(hits);
        }
        results
    }

    #[allow(clippy::suspicious_map)]
    pub fn leaf_search(&self, query: ArrayView1<f64>, radius: Option<f64>, clusters: ClusterResults) -> Results {
        let radius = radius.unwrap_or(0.);
        let results: Results = DashMap::new();
        clusters
            .par_iter()
            .map(|(cluster, _)| {
                let distances: Vec<f64> = cluster.indices
                    .par_iter()
                    .map(|&i| self.query_distance(self.dataset.row(i), query))
                    .collect();
                cluster.indices
                    .par_iter()
                    .zip(distances.par_iter())
                    .map(|(&i, &d)| if d <= radius {results.insert(i, d)} else {None})
                    .count();
            })
            .count();
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
