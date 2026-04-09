//! Trees built under Euclidean distance.

use std::collections::HashSet;

use abd_clam::{
    Cakes, PartitionStrategy, Tree,
    cakes::{MeasurableSearchQuality, Search},
    common_metrics,
};

use crate::{
    tree::data::{ShellDataType, npy::read_npy_generic as read_npy, shuffle_and_truncate},
    utils::SearchBencher,
};

use super::{ShellTree, TreeType};

/// Trees for Vector data under Euclidean distance.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EuclideanTree {
    /// Vectors are represented as `Vec<f64>`s and the distance values are `f64`s.
    F64(TreeType<usize, Vec<f64>, f64>),
    /// Vectors are represented as `Vec<f32>`s and the distance values are `f32`s.
    F32(TreeType<usize, Vec<f32>, f32>),
}

impl From<EuclideanTree> for ShellTree {
    fn from(euclidean_tree: EuclideanTree) -> Self {
        Self::Euclidean(euclidean_tree)
    }
}

impl databuf::Encode for EuclideanTree {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::F64(tree) => {
                b"F64".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
            Self::F32(tree) => {
                b"F32".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
        }
    }
}

impl<'de> databuf::Decode<'de> for EuclideanTree {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        // Decode the variant name first to determine which variant to decode.
        let variant = <[u8; 3]>::decode::<CONFIG>(buffer)?;
        match &variant {
            b"F64" => {
                let tree = Tree::decode::<CONFIG>(buffer)?;
                Ok(Self::F64(tree.with_metric(common_metrics::euclidean)))
            }
            b"F32" => {
                let tree = Tree::decode::<CONFIG>(buffer)?;
                Ok(Self::F32(tree.with_metric(common_metrics::euclidean)))
            }
            _ => Err(format!("Invalid variant for EuclideanTree: {variant:?}. Expected one of: F64, F32").into()),
        }
    }
}

impl EuclideanTree {
    /// Build the tree from the given data and data type.
    ///
    /// # Arguments
    ///
    /// * `data_path` - The path to the input data file.
    /// * `data_type` - The data type to use for building the tree.
    /// * `rng` - The random number generator to use.
    /// * `num_samples` - The number of samples to read from the input data file. If `None`, read all samples.
    /// * `strategy` - The partition strategy to use for building the tree.
    ///
    /// # Returns
    ///
    /// A tuple containing the built tree and a name generated for the output file based on the input data and data type.
    ///
    /// # Errors
    ///
    /// * If the input data file cannot be read or parsed.
    /// * If the data type is not compatible with the Euclidean metric.
    pub fn build<P: AsRef<std::path::Path>, R: rand::Rng>(
        data_path: &P,
        data_type: &ShellDataType,
        rng: &mut R,
        num_samples: Option<usize>,
        strategy: &PartitionStrategy,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match data_type {
            ShellDataType::F64 => {
                let data = read_npy::<_, f64>(data_path)?;
                let data = data.into_iter().enumerate().collect(); // Convert to (index, vector) pairs.
                let data = shuffle_and_truncate(data, rng, num_samples);
                let metric: fn(&_, &_) -> f64 = common_metrics::euclidean;
                let tree = Tree::par_new(data, metric, &|_| (), &|c| c.cardinality() > 2, strategy)?;
                Ok(Self::F64(tree))
            }
            ShellDataType::F32 => {
                let data = read_npy::<_, f32>(data_path)?;
                let data = data.into_iter().enumerate().collect(); // Convert to (index, vector) pairs.
                let data = shuffle_and_truncate(data, rng, num_samples);
                let metric: fn(&_, &_) -> f32 = common_metrics::euclidean;
                let tree = Tree::par_new(data, metric, &|_| (), &|c| c.cardinality() > 2, strategy)?;
                Ok(Self::F32(tree))
            }
            _ => Err(format!("Invalid data type for building a tree with the Euclidean metric: {data_type:?}. Expected one of: 'F64', 'F32'").into()),
        }
    }

    /// Searches the tree using the given algorithm and path to the query data.
    pub fn search<P: AsRef<std::path::Path>>(
        &self,
        algorithm: &str,
        queries_path: &P,
        num_queries: Option<usize>,
    ) -> Result<super::SearchHits, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::F64(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<f64>>()?;

                let queries = {
                    let mut queries = read_npy::<_, f64>(queries_path)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries
                };

                let hits = algorithm.par_batch_search(tree, &queries);
                Ok(super::SearchHits::F64(hits))
            }
            Self::F32(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<f32>>()?;

                let queries = {
                    let mut queries = read_npy::<_, f32>(queries_path)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries
                };

                let hits = algorithm.par_batch_search(tree, &queries);
                Ok(super::SearchHits::F32(hits))
            }
        }
    }

    /// Benchmark the algorithm using the given queries and quality metrics.
    pub fn bench<P: AsRef<std::path::Path>>(
        &self,
        algorithm: &str,
        queries_path: &P,
        num_queries: Option<usize>,
        quality_metrics: HashSet<MeasurableSearchQuality>,
    ) -> Result<SearchBencher, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::F64(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<f64>>()?;

                let queries = {
                    let mut queries = read_npy::<_, f64>(queries_path)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries
                };

                super::bench_tree(tree, &queries, &algorithm, quality_metrics)
            }
            Self::F32(tree) => {
                // Parse the algorithm string into a `Cakes` algorithm.
                let algorithm = algorithm.parse::<Cakes<f32>>()?;

                let queries = {
                    let mut queries = read_npy::<_, f32>(queries_path)?;
                    if let Some(num) = num_queries {
                        queries.truncate(num);
                    }
                    queries
                };

                super::bench_tree(tree, &queries, &algorithm, quality_metrics)
            }
        }
    }
}
