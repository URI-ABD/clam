//! Tree types supported in the CLAM Shell.

use core::borrow::Borrow;
use std::{collections::HashSet, path::Path};

use abd_clam::{
    Cakes, DistanceValue, PartitionStrategy, Tree,
    cakes::{MeasurableSearchQuality, Search},
};
use databuf::{Decode, Encode, config::DEFAULT as DATABUF_DEFAULT};

use crate::utils::{SearchBencher, SearchHits};

pub mod data;

pub mod cosine;
pub mod euclidean;
pub mod levenshtein;

/// The available metrics for the Shell CLI.
#[derive(clap::ValueEnum, Debug, Clone)]
#[non_exhaustive]
pub enum ShellMetric {
    /// The Levenshtein edit distance between two strings.
    #[clap(name = "levenshtein")]
    Levenshtein,
    /// The Euclidean distance between two vectors.
    #[clap(name = "euclidean")]
    Euclidean,
    /// The cosine distance between two vectors.
    #[clap(name = "cosine")]
    Cosine,
}

impl core::fmt::Display for ShellMetric {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let name = match self {
            Self::Levenshtein => "levenshtein",
            Self::Euclidean => "euclidean",
            Self::Cosine => "cosine",
        };
        write!(f, "{name}")
    }
}

/// A type alias for the different tree types supported in the CLAM Shell.
type TreeType<Id, I, T> = Tree<Id, I, T, (), fn(&I, &I) -> T>;

/// An enum of the different tree types supported in the CLAM Shell.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum ShellTree {
    /// Sequence data under Levenshtein distance.
    Levenshtein(levenshtein::LevenshteinTree),
    /// Vector data under Euclidean distance.
    Euclidean(euclidean::EuclideanTree),
    /// Vector data under Cosine distance.
    Cosine(cosine::CosineTree),
}

impl Encode for ShellTree {
    fn encode<const CONFIG: u16>(&self, buffer: &mut (impl std::io::Write + ?Sized)) -> std::io::Result<()> {
        match self {
            Self::Levenshtein(tree) => {
                b"Lev".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
            Self::Euclidean(tree) => {
                b"Euc".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
            Self::Cosine(tree) => {
                b"Cos".encode::<CONFIG>(buffer)?;
                tree.encode::<CONFIG>(buffer)
            }
        }
    }
}

impl<'de> Decode<'de> for ShellTree {
    fn decode<const CONFIG: u16>(buffer: &mut &'de [u8]) -> databuf::Result<Self> {
        // Decode the variant name first to determine which variant to decode.
        let variant = <[u8; 3]>::decode::<CONFIG>(buffer)?;
        match &variant {
            b"Lev" => levenshtein::LevenshteinTree::decode::<CONFIG>(buffer).map(Self::from),
            b"Euc" => euclidean::EuclideanTree::decode::<CONFIG>(buffer).map(Self::from),
            b"Cos" => cosine::CosineTree::decode::<CONFIG>(buffer).map(Self::from),
            _ => Err(format!("Invalid variant for ShellTree: {variant:?}. Expected one of: Lev, Euc, Cos").into()),
        }
    }
}

impl ShellTree {
    /// Build the tree from the given data and distance metric.
    ///
    /// # Arguments
    ///
    /// * `data_path` - The path to the input data file.
    /// * `metric` - The distance metric to use for building the tree.
    /// * `rng` - The random number generator to use.
    /// * `num_samples` - The number of samples to read from the input data file. If `None`, read all samples.
    /// * `strategy` - The partition strategy to use for building the tree.
    ///
    /// # Returns
    ///
    /// A tuple containing the built tree and a name generated for the output file based on the input data and metric.
    ///
    /// # Errors
    ///
    /// * If the input data file cannot be read or parsed.
    /// * If the specified metric is not compatible with the data format of the input data file.
    pub fn build<P: AsRef<Path>, R: rand::Rng>(
        data_path: &P,
        data_type: &data::ShellDataType,
        rng: &mut R,
        num_samples: Option<usize>,
        metric: &ShellMetric,
        strategy: &PartitionStrategy,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        match metric {
            ShellMetric::Levenshtein => levenshtein::LevenshteinTree::build(data_path, data_type, rng, num_samples, strategy).map(Self::from),
            ShellMetric::Euclidean => euclidean::EuclideanTree::build(data_path, data_type, rng, num_samples, strategy).map(Self::from),
            ShellMetric::Cosine => cosine::CosineTree::build(data_path, data_type, rng, num_samples, strategy).map(Self::from),
        }
    }

    /// Searches the tree using the given algorithm and path to the query data.
    pub fn search<P: AsRef<std::path::Path>>(
        &self,
        algorithm: &str,
        queries_path: &P,
        num_queries: Option<usize>,
    ) -> Result<SearchHits, Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Levenshtein(tree) => tree.search(algorithm, queries_path, num_queries),
            Self::Euclidean(tree) => tree.search(algorithm, queries_path, num_queries),
            Self::Cosine(tree) => tree.search(algorithm, queries_path, num_queries),
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
            Self::Levenshtein(tree) => tree.bench(algorithm, queries_path, num_queries, quality_metrics),
            Self::Euclidean(tree) => tree.bench(algorithm, queries_path, num_queries, quality_metrics),
            Self::Cosine(tree) => tree.bench(algorithm, queries_path, num_queries, quality_metrics),
        }
    }

    /// Writes the tree to the specified output directory with the specified name, using `databuf` for serialization.
    pub fn write<P: AsRef<Path>>(&self, out_path: &P) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let bytes = self.to_bytes::<DATABUF_DEFAULT>();
        std::fs::write(out_path, bytes).map(|()| out_path)?;
        Ok(())
    }

    /// Reads a tree from the specified input file, using `databuf` for deserialization.
    pub fn read<P: AsRef<Path>>(in_path: &P) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let bytes = std::fs::read(in_path)?;
        Self::from_bytes::<DATABUF_DEFAULT>(&bytes)
    }
}

/// Infers a name for the output tree file based on the input data path and metric.
pub fn infer_tree_name<P: AsRef<Path>>(
    data_path: &P,
    data_type: &data::ShellDataType,
    metric: &ShellMetric,
    strategy: &PartitionStrategy,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    data_path.as_ref().file_stem().map_or_else(
        || Err(format!("Data path does not have a file name: {:?}", data_path.as_ref()).into()),
        |data_name| {
            let name = data_name.to_string_lossy();
            Ok(format!("tree-{name}-{data_type}-{metric}-{strategy}.tree.bin"))
        },
    )
}

/// Benchmarks the given algorithm on the given tree and queries, and returns a summary of the results.
fn bench_tree<Id, I, T, A, M, Item, Query>(
    tree: &Tree<Id, Item, T, A, M>,
    queries: &[Query],
    algorithm: &Cakes<T>,
    quality_metrics: HashSet<MeasurableSearchQuality>,
) -> Result<SearchBencher, Box<dyn std::error::Error + Send + Sync>>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    A: Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
    Item: Borrow<I> + Send + Sync,
    Query: Borrow<I> + Send + Sync,
{
    ftlog::info!("Computing true neighbors using linear search for benchmark...");
    let linear_alg = algorithm.linear_variant();
    let true_neighbors = linear_alg.par_batch_search(tree, queries);

    ftlog::info!("Initializing benchmark...");
    let mut benchmark = SearchBencher::new(None, quality_metrics);

    // TODO: Run enough batches to get a statistically significant and stable benchmark result. This is currently set to 10 for testing purposes.
    let n_batches = 10;
    for i in 0..n_batches {
        ftlog::info!("Running batch {}/{n_batches} of benchmark...", i + 1);

        let start = std::time::Instant::now();
        let search_results = algorithm.par_batch_search(tree, queries);
        let duration = start.elapsed();

        ftlog::info!(
            "Registering batch {}/{n_batches} with duration {:.3e} seconds...",
            i + 1,
            duration.as_secs_f64()
        );
        benchmark.add_batch(duration, &search_results, &true_neighbors)?;
    }

    ftlog::info!("Benchmark completed with {n_batches} batches.");
    Ok(benchmark)
}
