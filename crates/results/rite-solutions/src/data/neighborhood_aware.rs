//! A `Dataset` in which every point stores the distances to its `k` nearest neighbors.

use abd_clam::{
    cluster::ParCluster, dataset::{metric_space::ParMetricSpace, ParDataset}, utils::{mean, standard_deviation}, Cluster, Dataset, FlatVec, Metric, MetricSpace, Permutable
};
use ftlog::debug;
use rayon::prelude::*;

use super::wasserstein::wasserstein;

type Fv = FlatVec<Vec<f32>, f32, usize>;

pub struct NeighborhoodAwareScore{
    pub score: f32,
    pub standard_deviation: f32
}

/// A `Dataset` in which every point stores the distances to its `k` nearest neighbors.
#[allow(clippy::type_complexity)]
pub struct NeighborhoodAware {
    data: FlatVec<Vec<f32>, f32, (usize, Vec<(usize, f32)>)>,
    k: usize,
}

#[allow(dead_code)]
impl NeighborhoodAware {
    /// Create a new `NeighborhoodAware` `Dataset`.
    ///
    /// This will run knn-search on every point in the dataset and store the
    /// results in the dataset.
    pub fn new<C: Cluster<Vec<f32>, f32, Fv>>(data: &Fv, root: &C, k: usize) -> Self {
        let alg = abd_clam::cakes::Algorithm::KnnLinear(k + 1);

        let results: Vec<(usize, Vec<(usize, f32)>)> = data
            .instances()
            .iter()
            .enumerate()
            .map(|(_, query)| {
                let mut neighbors = alg.search(data, root, query);
                neighbors.sort_by(|&(_, a),(_, b)|a.partial_cmp(b).unwrap());
                neighbors
            })
            .zip(data.metadata().iter())
            .map(|(h, &i)| (i, h))
            .collect();
        
        let data = data
            .clone()
            .with_metadata(results)
            .unwrap_or_else(|e| unreachable!("We created the correct size for neighborhood aware data: {e}"));
        Self { data, k }
    }

    /// Parallel version of `new`.
    pub fn par_new<C: ParCluster<Vec<f32>, f32, Fv>>(data: &Fv, root: &C, k: usize) -> Self {
        let alg = abd_clam::cakes::Algorithm::KnnLinear(k + 1);

        let results: Vec<(usize, Vec<(usize, f32)>)> = data
            .instances()
            .par_iter()
            .map(|query| {
                let mut neighbors = alg.par_search(data, root, query);
                neighbors.par_sort_by(|&(_, a),(_, b)|a.partial_cmp(b).unwrap());
                neighbors
            })
            .zip(data.metadata().par_iter())
            .map(|(h, &i)| (i, h))
            .collect();
        
        let data = data
            .clone()
            .with_metadata(results)
            .unwrap_or_else(|e| unreachable!("We created the correct size for neighborhood aware data: {e}"));
        Self { data, k }
    }
    
    pub fn outlier_score<C: Cluster<Vec<f32>, f32, Self>>(&self, root: &C, query: &Vec<f32>) -> NeighborhoodAwareScore {
        debug!("Entering is_outlier.");
        
        let alg = abd_clam::cakes::Algorithm::KnnLinear(self.k);
        
        let hits = alg.search(self, root, query);
        
        debug!("");
        debug!("Hits: {:?}", hits);
        
        let neighbors_distances = hits
            .iter()
            .map(|&(i, _)| {
                self.neighbor_distances(i)
            })
            .collect::<Vec<_>>();
        
        let neighbor_wass_dist_mat = neighbors_distances.iter().map(|v| {
            neighbors_distances.iter().map(|q| wasserstein(v, q)).collect::<Vec<f32>>()
        }).collect::<Vec<Vec<f32>>>();
        
        debug!("Wasserstein distance matrix of neighbors:");
        for a in &neighbor_wass_dist_mat{
            debug!("\t{:?}", *a);
        }
        
        let query_distances = hits.iter().map(|&(_, d)| d).collect::<Vec<_>>();
        
        let query_wass_distances = neighbors_distances.iter().map(|v|{
            wasserstein(&query_distances, v)
        }).collect::<Vec<f32>>();
        
        debug!("");
        debug!("Query wasserstein distances");
        debug!("{:?}", query_wass_distances);
        
        /*
        Now that we have the wasserstein distances, we need to collapse
        the vectors into some individual score. The way I am choosing to do
        this is:
         - find the means of each of the neighbor groups (Collapsing the Vec<Vec<f32>> into a Vec<f32>)
         - find the standard deviation of the neighbor distances
         - using this standard deviation, find the average Z-score of the query against the neighbors
        */
        
        let neighbor_means = neighbor_wass_dist_mat.iter()
            .map(|v: &Vec<f32>|{
                mean::<f32, f32>(v)
            })
            .collect::<Vec<_>>();
        
        debug!("");
        debug!("Neighbor means:");
        debug!("{:?}", neighbor_means);
        
        let mean_of_neighbor_means: f32 = mean(&neighbor_means);
        
        let neighbor_standard_deviation: f32 = standard_deviation(&neighbor_means);
        
        // let mean_squeared_errors = neighbor_means.iter()
        //     .zip(query_wass_distances.iter())
        //     .map(|(&u, &x)| (x - u).powi(2))
        //     .collect::<Vec<_>>();
        
        let mean_squeared_errors = query_wass_distances.iter()
            .map(|&x| (x - mean_of_neighbor_means).powi(2))
            .collect::<Vec<_>>();
        
        debug!("");
        debug!("Mean squared errors:");
        debug!("{:?}", mean_squeared_errors);
        
        let average_mean_squared_error: f32 = mean(&mean_squeared_errors);
        
        NeighborhoodAwareScore{
            score: average_mean_squared_error,
            standard_deviation: neighbor_standard_deviation,
        }
    }
    
    /// Check if a point is an outlier.
    pub fn is_outlier<C: Cluster<Vec<f32>, f32, Self>>(&self, root: &C, query: &Vec<f32>) -> bool {
        let NeighborhoodAwareScore {
            score,
            standard_deviation 
        } = self.outlier_score(root, query);
        
        let sigma_range = 2f32 * standard_deviation;
        
        debug!("Score: {score}, standard deviation: {standard_deviation}");
        
        score.abs() > sigma_range
    }

    /// Get the distances to the `k` nearest neighbors of a point.
    // fn neighbor_distances(&self, i: usize) -> Vec<f32> {
    //     self.data.metadata()[i].1.iter().map(|&(_, d)| d).collect()
    // }
    
    /// Get the nearest neighbors not including the queried index.
    fn neighbor_distances(&self, i: usize) -> Vec<f32> {
        self.data.metadata()[i].1.iter()
            .filter(|(ind, _)| *ind != i)
            .map(|&(_, d)| d).collect()
    }
}

impl MetricSpace<Vec<f32>, f32> for NeighborhoodAware {
    fn metric(&self) -> &Metric<Vec<f32>, f32> {
        self.data.metric()
    }

    fn set_metric(&mut self, metric: Metric<Vec<f32>, f32>) {
        self.data.set_metric(metric);
    }
}

impl Dataset<Vec<f32>, f32> for NeighborhoodAware {
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

impl ParMetricSpace<Vec<f32>, f32> for NeighborhoodAware {}

impl ParDataset<Vec<f32>, f32> for NeighborhoodAware {}
