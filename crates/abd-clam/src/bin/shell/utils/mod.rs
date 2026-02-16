//! Helpers for the CLAM Shell.

use std::{collections::HashSet, path::PathBuf};

use abd_clam::{
    DistanceValue,
    cakes::{
        MeasurableSearchQuality,
        quality::{Recall, RelativeDistanceError, SearchQuality},
    },
    utils::MeasuredQuality,
};
use ftlog::{
    LevelFilter, LoggerGuard,
    appender::{FileAppender, Period},
};

mod reports;

pub use reports::ReportFormat;

/// An enum to account for the types of distance values that can be used in search results.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode)]
#[non_exhaustive]
pub enum SearchHits {
    /// Search hits with `usize` distance values (e.g., edit distance).
    Usize(Vec<Vec<(usize, usize)>>),
    /// Search hits with `f64` distance values, (e.g., euclidean, cosine, etc.).
    F64(Vec<Vec<(usize, f64)>>),
    /// Search hits with `f32` distance values, (e.g., euclidean, cosine, etc.).
    F32(Vec<Vec<(usize, f32)>>),
}

/// Summary of the benchmarks for a search algorithm, including the duration and quality of each batch of search queries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode)]
#[must_use]
pub struct SearchBenchSummary {
    /// The total number of batches of search queries.
    num_batches: usize,
    /// The total number of queries across all batches.
    total_num_queries: usize,
    /// The total duration of all batches (in seconds).
    total_duration: f64,
    /// The mean throughput (queries per second) across all batches.
    mean_throughput: f64,
    /// The standard deviation of the throughput across all batches.
    throughput_std_dev: f64,
    /// The aggregate recall across all batches, if measured.
    aggregate_recall: Option<MeasuredQuality<Recall>>,
    /// The aggregate relative distance error across all batches, if measured.
    aggregate_relative_distance_error: Option<MeasuredQuality<RelativeDistanceError>>,
}

impl core::fmt::Display for SearchBenchSummary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Search Benchmark Summary:")?;
        writeln!(f, "  Number of batches: {}", self.num_batches)?;
        writeln!(f, "  Total number of queries: {}", self.total_num_queries)?;
        writeln!(f, "  Total duration: {:.2?}", self.total_duration)?;
        writeln!(f, "  Mean throughput (queries/sec): {:.2}", self.mean_throughput)?;
        writeln!(f, "  Throughput standard deviation (queries/sec): {:.2}", self.throughput_std_dev)?;
        if let Some(aggregate_recall) = &self.aggregate_recall {
            writeln!(f, "  Aggregate {aggregate_recall:.4}")?;
        } else {
            writeln!(f, "  Aggregate Recall: N/A")?;
        }
        if let Some(aggregate_relative_distance_error) = &self.aggregate_relative_distance_error {
            writeln!(f, "  Aggregate {aggregate_relative_distance_error:.4}")?;
        } else {
            writeln!(f, "  Aggregate Relative Distance Error: N/A")?;
        }
        Ok(())
    }
}

/// Benchmarks of search algorithms.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
pub struct SearchBencher {
    /// The benchmark for each batch of search queries.
    batches: Vec<Batch>,
    /// The qualities that will be measured for each batch, if applicable.
    qualities: Vec<MeasurableSearchQuality>,
}

impl SearchBencher {
    /// Creates a new `SearchBencher` instance, ready to record benchmarks for search algorithms.
    pub fn new(num_expected_batches: Option<usize>, qualities: HashSet<MeasurableSearchQuality>) -> Self {
        let batches = num_expected_batches.map_or_else(Vec::new, Vec::with_capacity);
        let qualities = qualities.into_iter().collect();
        Self { batches, qualities }
    }

    /// Records the duration of a batch of search queries.
    ///
    /// # Arguments
    ///
    /// - `duration`: The time taken to execute the batch of search queries.
    /// - `search_results`: The search results for the batch of queries.
    /// - `true_neighbors`: The true neighbors for the batch of queries, used to measure the quality of the search results.
    ///
    /// # Errors
    ///
    /// - If the batch of search results is empty.
    /// - If the number of search results does not match the number of true neighbors, these should both be equal to the number of queries in the batch.
    pub fn add_batch<T: DistanceValue>(
        &mut self,
        duration: std::time::Duration,
        search_results: &[Vec<(usize, T)>],
        true_neighbors: &[Vec<(usize, T)>],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if search_results.is_empty() {
            return Err("Batch cannot be empty".into());
        }
        if search_results.len() != true_neighbors.len() {
            return Err(format!(
                "Search results and true neighbors must have the same number of queries, but got {} and {} respectively",
                search_results.len(),
                true_neighbors.len()
            )
            .into());
        }

        let num_queries = search_results.len();
        let qualities = self
            .qualities
            .iter()
            .map(|quality| quality.measure_batch(search_results, true_neighbors))
            .inspect(|q| ftlog::info!("  Registered batch quality: {q}"))
            .collect();

        self.batches.push(Batch {
            num_queries,
            duration,
            qualities,
        });

        Ok(())
    }

    /// Aggregates the qualities of all batches into a single measurement for each quality metric, if applicable.
    pub fn summarize(self) -> SearchBenchSummary {
        Batch::summarize(self.batches)
    }
}

/// The benchmark for a single batch.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[must_use]
struct Batch {
    /// The number of queries in the batch.
    num_queries: usize,
    /// The duration of the batch.
    duration: std::time::Duration,
    /// The quality of the batch, if applicable.
    qualities: Vec<SearchQuality>,
}

impl Batch {
    /// Aggregates the qualities of several batches into a summary.
    #[expect(clippy::cast_precision_loss)]
    fn summarize(batches: Vec<Self>) -> SearchBenchSummary {
        let num_batches = batches.len();
        let total_num_queries = batches.iter().map(|batch| batch.num_queries).sum();
        let total_duration = batches.iter().map(|batch| batch.duration).sum::<std::time::Duration>().as_secs_f64();

        let throughputs: Vec<f64> = batches
            .iter()
            .map(|batch| batch.num_queries as f64 / batch.duration.as_secs_f64())
            .collect::<Vec<_>>();
        let mean_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let throughput_std_dev = (throughputs.iter().map(|throughput| (throughput - mean_throughput).powi(2)).sum::<f64>() / throughputs.len() as f64).sqrt();

        let measurements = batches.into_iter().flat_map(|batch| batch.qualities).collect();
        let (aggregate_recall, aggregate_relative_distance_error) = SearchQuality::aggregate_batch(measurements);

        SearchBenchSummary {
            num_batches,
            total_num_queries,
            total_duration,
            mean_throughput,
            throughput_std_dev,
            aggregate_recall,
            aggregate_relative_distance_error,
        }
    }
}

/// Checks if the given path exists, and some other properties of it depending on the input parameters.
///
/// # Arguments
///
/// - `path`: The path to check.
/// - `is_dir`: Whether the path should be a directory (as opposed to a file).
/// - `create_if_not_exists`: Whether to create the path if it does not exist.
///
/// # Errors
///
/// - If the path could not be canonicalized.
/// - If the path does not exist and `create_if_not_exists` is false.
/// - If the path does not have the expected type (file vs directory).
/// - If there was an error creating the path when it did not exist and `create_if_not_exists` is true.
pub fn check_path_exists<P: AsRef<std::path::Path>>(
    path: &P,
    is_dir: bool,
    create_if_not_exists: bool,
) -> Result<PathBuf, Box<dyn std::error::Error + Send + Sync>> {
    let path = path.as_ref();
    if !path.exists() {
        if create_if_not_exists {
            if is_dir {
                std::fs::create_dir_all(path)?;
            } else {
                std::fs::File::create(path)?;
            }
        } else {
            return Err(format!("Path {path:?} does not exist and was not requested to be created").into());
        }
    }

    let path = path.canonicalize().map_err(|e| format!("Failed to canonicalize path {path:?}: {e}"))?;

    if is_dir && !path.is_dir() {
        return Err(format!("Path {path:?} is not a directory").into());
    }

    if !is_dir && !path.is_file() {
        return Err(format!("Path {path:?} is not a file").into());
    }

    Ok(path)
}

/// Configures the logger.
///
/// # Errors
///
/// - If a logs directory could not be located/created.
/// - If the logger could not be initialized.
pub fn configure_logger(file_name: &str) -> Result<(LoggerGuard, PathBuf), Box<dyn std::error::Error + Send + Sync>> {
    let logs_dir = check_path_exists(&PathBuf::from(".").join("logs"), true, true)?;
    let log_path = logs_dir.join(file_name);

    let err_stem = log_path
        .file_stem()
        .ok_or_else(|| format!("Failed to get file stem for {log_path:?}"))?
        .to_str()
        .ok_or_else(|| format!("Failed to convert file stem to string for {log_path:?}"))?;
    let err_stem = format!("{err_stem}-err");
    let err_path = log_path.with_file_name(err_stem);

    let writer = FileAppender::builder().path(&log_path).rotate(Period::Day).build();
    let guard = ftlog::Builder::new()
        // global max log level
        .max_log_level(LevelFilter::Info)
        // define root appender, pass None would write to stderr
        .root(writer)
        // write `Warn` and `Error` logs in ftlog::appender to `err_path` instead of `log_path`
        .filter("ftlog::appender", "ftlog-appender", LevelFilter::Warn)
        .appender("ftlog-appender", FileAppender::new(err_path))
        .try_init()
        .map_err(|e| format!("Failed to initialize logger: {e}"))?;

    Ok((guard, log_path))
}
