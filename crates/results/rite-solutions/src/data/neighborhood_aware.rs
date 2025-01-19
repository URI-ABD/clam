//! A `Dataset` in which every point stores the distances to its `k` nearest neighbors.

use abd_clam::{
    cakes::{self, ParSearchAlgorithm, ParSearchable, SearchAlgorithm, Searchable},
    cluster::ParCluster,
    dataset::{AssociatesMetadata, AssociatesMetadataMut, ParDataset, Permutable},
    metric::ParMetric,
    Cluster, Dataset, FlatVec, Metric,
};
use rayon::prelude::*;

use super::wasserstein;

type Fv = FlatVec<Vec<f32>, usize>;

/// A `Dataset` in which every point stores the distances to its `k` nearest neighbors.
#[allow(clippy::type_complexity)]
pub struct NeighborhoodAware {
    data: FlatVec<Vec<f32>, (usize, Vec<(usize, f32)>)>,
    k: usize,
}

impl NeighborhoodAware {
    /// Create a new `NeighborhoodAware` `Dataset`.
    ///
    /// This will run knn-search on every point in the dataset and store the
    /// results in the dataset.
    pub fn new<C: Cluster<f32>, M: Metric<Vec<f32>, f32>>(data: &Fv, metric: &M, root: &C, k: usize) -> Self {
        let alg = cakes::KnnLinear(k);

        let results = data
            .items()
            .iter()
            .map(|query| alg.search(data, metric, root, query))
            .zip(data.metadata().iter())
            .map(|(h, &i)| (i, h))
            .collect::<Vec<_>>();

        let data = data
            .clone()
            .with_metadata(&results)
            .unwrap_or_else(|e| unreachable!("We created the correct size for neighborhood aware data: {e}"));
        Self { data, k }
    }

    /// Parallel version of `new`.
    pub fn par_new<C: ParCluster<f32>, M: ParMetric<Vec<f32>, f32>>(data: &Fv, metric: &M, root: &C, k: usize) -> Self {
        let alg = cakes::KnnLinear(k);

        let results = data
            .items()
            .par_iter()
            .map(|query| alg.par_search(data, metric, root, query))
            .zip(data.metadata().par_iter())
            .map(|(h, &i)| (i, h))
            .collect::<Vec<_>>();

        let data = data
            .clone()
            .with_metadata(&results)
            .unwrap_or_else(|e| unreachable!("We created the correct size for neighborhood aware data: {e}"));
        Self { data, k }
    }

    /// Check if a point is an outlier.
    pub fn is_outlier<C: Cluster<f32>, M: Metric<Vec<f32>, f32>>(
        &self,
        metric: &M,
        root: &C,
        query: &Vec<f32>,
        threshold: f32,
    ) -> bool {
        let alg = cakes::KnnLinear(self.k);

        let hits = alg.search(self, metric, root, query);
        let neighbors_distances = hits
            .iter()
            .map(|&(i, _)| self.neighbor_distances(i))
            .collect::<Vec<_>>();

        // TODO: Compute all-pairs matrix of wasserstein distances among the neighbors' distance distributions.

        // TODO: Compute the wasserstein distances for the query

        // TODO: Optionally use the threshold to determine if the query is an outlier.

        let distances = hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
        // TODO: The rest of this is wrong.
        let wasserstein_distances = neighbors_distances
            .iter()
            .map(|d| wasserstein::wasserstein(d, &distances))
            .collect::<Vec<_>>();
        let mean_wasserstein: f32 = abd_clam::utils::mean(&wasserstein_distances);

        mean_wasserstein > threshold
    }

    /// Get the distances to the `k` nearest neighbors of a point.
    fn neighbor_distances(&self, i: usize) -> Vec<f32> {
        self.data.metadata()[i].1.iter().map(|&(_, d)| d).collect()
    }
}

impl Dataset<Vec<f32>> for NeighborhoodAware {
    fn name(&self) -> &str {
        self.data.name()
    }

    fn with_name(self, name: &str) -> Self {
        Self {
            data: self.data.with_name(name),
            k: self.k,
        }
    }

    fn cardinality(&self) -> usize {
        self.data.cardinality()
    }

    fn dimensionality_hint(&self) -> (usize, Option<usize>) {
        self.data.dimensionality_hint()
    }

    fn get(&self, index: usize) -> &Vec<f32> {
        self.data.get(index)
    }
}

impl Permutable for NeighborhoodAware {
    fn permutation(&self) -> Vec<usize> {
        self.data.permutation()
    }

    fn set_permutation(&mut self, permutation: &[usize]) {
        self.data.set_permutation(permutation);
    }

    fn swap_two(&mut self, i: usize, j: usize) {
        self.data.swap_two(i, j);
    }
}

impl ParDataset<Vec<f32>> for NeighborhoodAware {}

impl<C: Cluster<f32>, M: Metric<Vec<f32>, f32>> Searchable<Vec<f32>, f32, C, M> for NeighborhoodAware {
    fn query_to_center(&self, metric: &M, query: &Vec<f32>, cluster: &C) -> f32 {
        self.data.query_to_center(metric, query, cluster)
    }

    fn query_to_all(&self, metric: &M, query: &Vec<f32>, cluster: &C) -> impl Iterator<Item = (usize, f32)> {
        self.data.query_to_all(metric, query, cluster)
    }
}

impl<C: ParCluster<f32>, M: ParMetric<Vec<f32>, f32>> ParSearchable<Vec<f32>, f32, C, M> for NeighborhoodAware {
    fn par_query_to_center(&self, metric: &M, query: &Vec<f32>, cluster: &C) -> f32 {
        self.data.par_query_to_center(metric, query, cluster)
    }

    fn par_query_to_all(&self, metric: &M, query: &Vec<f32>, cluster: &C) -> impl ParallelIterator<Item = (usize, f32)> {
        self.data.par_query_to_all(metric, query, cluster)
    }
}
