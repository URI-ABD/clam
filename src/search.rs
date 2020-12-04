use std::{fmt, result};
use std::borrow::Borrow;
use std::sync::Arc;

use dashmap::{DashMap, DashSet};
use ndarray::ArrayView1;
use rayon::prelude::*;

use crate::cluster::Cluster;
use crate::criteria;
use crate::dataset::Dataset;
use crate::metric::Metric;
use crate::types::{Index, Indices};

type ClusterResults = Arc<DashSet<Arc<Cluster>>>;
type Results = Arc<DashMap<Index, f64>>;

pub struct Search {
    pub dataset: Arc<Dataset>,
    root: Arc<Cluster>,
    function: fn(ArrayView1<f64>, ArrayView1<f64>) -> f64,
}

impl fmt::Debug for Search {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Search")
            .field("dataset", &self.dataset)
            .finish()
    }
}

impl Search {
    // TODO: Add save and load methods with serde.
    pub fn build(dataset: Arc<Dataset>, max_depth: Option<usize>) -> Search {
        let criteria = match max_depth {
            Some(d) => vec![criteria::MaxDepth::new(d)],
            None => vec![],
        };
        let root = Cluster::new(
            Arc::clone(&dataset),
            "".to_string(),
            dataset.indices(),
        ).partition(&criteria);
        Search {
            dataset: Arc::clone(&dataset),
            root: Arc::new(root),
            function: Metric::on_float(dataset.metric).unwrap(),
        }
    }

    pub fn diameter(&self) -> f64 { 2. * self.root.radius }

    pub fn indices(&self) -> Indices { self.dataset.indices() }

    pub fn rnn(&self, query: ArrayView1<f64>, radius: Option<f64>) -> Results {
        self.leaf_search(query, radius, self.tree_search(query, radius))
    }

    pub fn tree_search(&self, query: ArrayView1<f64>, radius: Option<f64>) -> ClusterResults {
        let radius = radius.unwrap_or(0.);
        let results: ClusterResults = Arc::new(DashSet::new());
        self._tree_search(&self.root, query, radius, Arc::clone(&results));
        results
    }

    fn _tree_search(&self, cluster: &Arc<Cluster>, query: ArrayView1<f64>, radius: f64, results: ClusterResults){
        match cluster.children.borrow() {
            Some((left, right)) => {
                rayon::join(
                    || if self.query_distance(query, left.argcenter) > (radius + left.radius) { }
                    else { self._tree_search(&left, query, radius, Arc::clone(&results)) },
                    || if self.query_distance(query, right.argcenter) > (radius + right.radius) { }
                    else { self._tree_search(&right, query, radius, Arc::clone(&results)) },
                );
            }
            None => { results.insert(Arc::clone(cluster)); },
        };
    }

    #[allow(clippy::suspicious_map)]
    pub fn leaf_search(&self, query: ArrayView1<f64>, radius: Option<f64>, clusters: ClusterResults) -> Results {
        let indices = clusters.iter()
            .map(|c| c.indices.clone())
            .into_iter()
            .flatten()
            .collect::<Indices>();
        self.linear_search(query, radius, Some(indices))
    }

    #[allow(clippy::suspicious_map)]
    pub fn linear_search(&self, query: ArrayView1<f64>, radius: Option<f64>, indices: Option<Indices>) -> Results {
        let radius = radius.unwrap_or(0.);
        let indices = indices.unwrap_or_else(|| self.dataset.indices());
        let distances = self.query_distances_from(query, &indices);
        let results: Results = Arc::new(DashMap::new());
        indices.par_iter()
            .zip(distances.par_iter())
            .map(|(&i, &d)| {
                if d <= radius { results.insert(i, d) }
                else { None }
            })
            .count();
        results
    }

    #[allow(clippy::ptr_arg)]
    fn query_distances_from(&self, query: ArrayView1<f64>, indices: &Indices) -> Vec<f64> {
        indices.par_iter().map(|&i| self.query_distance(query, i)).collect()
    }

    fn query_distance(&self, query: ArrayView1<f64>, index: Index) -> f64 {
        (self.function)(query, self.dataset.row(index))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::dataset::Dataset;
    use crate::utils::{DATASETS, read_data};

    use super::Search;

    #[test]
    fn test_search() {
        let name = DATASETS[0];
        let (data, _) = read_data(name).unwrap();
        let dataset = Arc::new(Dataset::new(data, "euclidean", true).unwrap());

        let search = Search::build(Arc::clone(&dataset), Some(25));

        assert_eq!(search.indices().len(), search.dataset.nrows());

        let search_str = [
            "Search { ".to_string(),
            format!("dataset: {:?}", dataset),
            " }".to_string(),
        ].join("");
        assert_eq!(format!("{:?}", search), search_str);

        let radius = Some(search.diameter() / 200.);

        for &q in dataset.indices()[0..30].iter() {
            let mut missed = false;
            let mut extra = false;
            let query = dataset.row(q);

            search.dataset.clear_cache();
            let naive_results = search.linear_search(query, radius, None);
            let naive_count = search.dataset.cache_size();

            search.dataset.clear_cache();
            let chess_results = search.rnn(query, radius);
            let chess_count = search.dataset.cache_size();

            for (i, _) in (*chess_results).clone() {
                if !naive_results.contains_key(&i) { extra = true; }
            }
            for (i, _) in (*naive_results).clone() {
                if !chess_results.contains_key(&i) { missed = true; }
            }
            assert!(!extra, "had some extras");
            assert!(!missed, "had some misses");
            assert!(chess_count <= naive_count, "chess should call distance function less often than naive")
        }
    }
}
