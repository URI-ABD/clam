//! CAKES search with a single shard.

use core::cmp::Ordering;

use std::path::Path;

use distances::Number;
use rayon::prelude::*;

use crate::{cakes::knn, cakes::rnn, Cluster, Dataset, Instance, PartitionCriterion, Tree, UniBall};

use super::Search;

/// CLAM-Accelerated K-nearest-neighbor Entropy-scaling Search.
///
/// The search time scales by the metric entropy of the dataset.
///
/// # Type Parameters
///
/// * `T` - The type of the instances.
/// * `U` - The type of the distance value.
/// * `D` - The type of the dataset.
#[derive(Debug)]
pub struct SingleShard<I: Instance, U: Number, D: Dataset<I, U>> {
    /// The tree used for the search.
    tree: Tree<I, U, D, UniBall<U>>,
    /// Best rnn-search algorithm.
    best_rnn: Option<rnn::Algorithm>,
    /// Best knn-search algorithm.
    best_knn: Option<knn::Algorithm>,
}

impl<I: Instance, U: Number, D: Dataset<I, U>> SingleShard<I, U, D> {
    /// Creates a new CAKES instance.
    ///
    /// # Arguments
    ///
    /// * `data` - The dataset to search.
    /// * `seed` - The seed to use for the random number generator.
    /// * `criteria` - The criteria to use for partitioning the tree.
    pub fn new<P: PartitionCriterion<U>>(data: D, seed: Option<u64>, criteria: &P) -> Self {
        Self {
            tree: Tree::new(data, seed).partition(criteria, seed),
            best_rnn: None,
            best_knn: None,
        }
    }

    /// Returns a reference to the dataset.
    pub const fn data(&self) -> &D {
        self.tree.data()
    }

    /// Returns a reference to the tree.
    pub const fn tree(&self) -> &Tree<I, U, D, UniBall<U>> {
        &self.tree
    }

    /// A helper function for sampling query indices for tuning.
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth in the tree to sample at.
    ///
    /// # Returns
    ///
    /// A vector of indices of cluster centers at the given depth.
    fn sample_query_indices(&self, depth: usize) -> Vec<usize> {
        self.tree
            .root()
            .subtree()
            .into_iter()
            .filter(|&c| c.depth() == depth || c.is_leaf() && c.depth() < depth)
            .map(Cluster::arg_center)
            .collect()
    }
}

impl<I: Instance, U: Number, D: Dataset<I, U>> Search<I, U, D> for SingleShard<I, U, D> {
    #[allow(clippy::similar_names)]
    fn save(&self, path: &Path) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("The path '{}' does not exist.", path.display()));
        }

        if !path.is_dir() {
            return Err(format!("The path '{}' is not a directory.", path.display()));
        }

        let tree_dir = path.join("tree");
        if !tree_dir.exists() {
            std::fs::create_dir(&tree_dir).map_err(|e| e.to_string())?;
        }
        self.tree.save(&tree_dir)?;

        let best_rnn = self
            .best_rnn
            .map_or_else(|| "None".to_string(), |a| a.name().to_string());
        let best_knn = self
            .best_knn
            .map_or_else(|| "None".to_string(), |a| a.name().to_string());

        let best_algo_file = path.join("best-algo.txt");
        std::fs::write(best_algo_file, format!("{best_rnn}\n{best_knn}")).map_err(|e| e.to_string())?;

        Ok(())
    }

    #[allow(clippy::similar_names)]
    fn load(path: &Path, metric: fn(&I, &I) -> U, is_expensive: bool) -> Result<Self, String>
    where
        Self: Sized,
    {
        if !path.exists() {
            return Err(format!("The path '{}' does not exist.", path.display()));
        }

        if !path.is_dir() {
            return Err(format!("The path '{}' is not a directory.", path.display()));
        }

        let best_algo_file = path.join("best-algo.txt");
        if !best_algo_file.exists() {
            return Err(format!("The file '{}' does not exist.", best_algo_file.display()));
        }

        let contents = std::fs::read_to_string(&best_algo_file).map_err(|e| e.to_string())?;
        let mut lines = contents.lines();
        let best_rnn = lines.next().ok_or_else(|| "The file is empty.".to_string())?;
        let best_knn = lines.next().ok_or_else(|| "The file is empty.".to_string())?;

        if lines.next().is_some() {
            return Err("The file has too many lines.".to_string());
        }

        let best_rnn = if best_rnn == "None" {
            None
        } else {
            Some(rnn::Algorithm::from_name(best_rnn)?)
        };

        let best_knn = if best_knn == "None" {
            None
        } else {
            Some(knn::Algorithm::from_name(best_knn)?)
        };

        let tree_dir = path.join("tree");
        let tree = Tree::<I, U, D, UniBall<_>>::load(&tree_dir, metric, is_expensive)?;

        Ok(Self {
            tree,
            best_rnn,
            best_knn,
        })
    }

    fn num_shards(&self) -> usize {
        1
    }

    fn shard_cardinalities(&self) -> Vec<usize> {
        vec![self.tree.data().cardinality()]
    }

    fn auto_tune_rnn(&mut self, radius: U, tuning_depth: usize) {
        let queries = self
            .sample_query_indices(tuning_depth)
            .into_iter()
            .map(|i| &self.data()[i])
            .collect::<Vec<_>>();

        (self.best_rnn, _, _) = rnn::Algorithm::variants()
            .iter()
            .map(|&algo| {
                let start = std::time::Instant::now();
                let hits = queries
                    .par_iter()
                    .map(|query| self.rnn_search(query, radius, algo))
                    .collect::<Vec<_>>();
                let elapsed = start.elapsed().as_secs_f32();
                (Some(algo), hits, elapsed)
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|| unreachable!("There are several variants of rnn-search."));
    }

    fn tuned_rnn_algorithm(&self) -> rnn::Algorithm {
        self.best_rnn.unwrap_or_default()
    }

    fn rnn_search(&self, query: &I, radius: U, algo: rnn::Algorithm) -> Vec<(usize, U)> {
        algo.search(query, radius, &self.tree)
    }

    fn linear_rnn_search(&self, query: &I, radius: U) -> Vec<(usize, U)> {
        self.rnn_search(query, radius, rnn::Algorithm::Linear)
    }

    fn auto_tune_knn(&mut self, k: usize, tuning_depth: usize) {
        let queries = self
            .sample_query_indices(tuning_depth)
            .into_iter()
            .map(|i| &self.data()[i])
            .collect::<Vec<_>>();

        (self.best_knn, _, _) = knn::Algorithm::variants()
            .iter()
            .map(|&algo| {
                let start = std::time::Instant::now();
                let hits = queries
                    .par_iter()
                    .map(|query| self.knn_search(query, k, algo))
                    .collect::<Vec<_>>();
                let elapsed = start.elapsed().as_secs_f32();
                (Some(algo), hits, elapsed)
            })
            .min_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .unwrap_or_else(|| unreachable!("There are several variants of knn-search."));
    }

    fn tuned_knn_algorithm(&self) -> knn::Algorithm {
        self.best_knn.unwrap_or_default()
    }

    fn knn_search(&self, query: &I, k: usize, algo: knn::Algorithm) -> Vec<(usize, U)> {
        algo.search(&self.tree, query, k)
    }

    fn linear_knn_search(&self, query: &I, k: usize) -> Vec<(usize, U)> {
        self.knn_search(query, k, knn::Algorithm::Linear)
    }
}
