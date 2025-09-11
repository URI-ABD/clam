//! Search a given tree for some given queries.

use std::fs;
use std::path::Path;

use abd_clam::cakes::Searchable;
use abd_clam::{Cluster, FlatVec, Metric};
use distances::Number;
use serde::{Deserialize, Serialize};

use crate::trees::{ShellBall, ShellPermutedBall};
use crate::{data::ShellFlatVec, metrics::ShellMetric, trees::ShellTree};

/// Represents the complete search results for all queries and algorithms.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResults {
    /// Results for each query
    pub queries: Vec<QueryResult>,
}

/// Represents the result for a single query across all algorithms.
#[derive(Debug, Serialize, Deserialize)]
pub struct QueryResult {
    /// Index of the query in the instances dataset
    pub query_index: usize,
    /// Results from each algorithm
    pub algorithms: Vec<AlgorithmResult>,
}

/// Represents the result from a single algorithm for a query.
#[derive(Debug, Serialize, Deserialize)]
pub struct AlgorithmResult {
    /// Name of the algorithm used
    pub algorithm: String,
    /// List of (index, distance) pairs for the neighbors found
    pub neighbors: Vec<(usize, f64)>,
}

/// Supported output formats based on file extension.
#[derive(Debug, Clone, Copy)]
enum OutputFormat {
    Json,
    Yaml,
}

impl OutputFormat {
    /// Determine format from file extension.
    fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path = path.as_ref();
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => Ok(OutputFormat::Json),
            Some("yaml") | Some("yml") => Ok(OutputFormat::Yaml),
            Some(ext) => Err(format!(
                "Unsupported file extension: '.{ext}'. Supported formats: .json, .yaml, .yml",
            )),
            None => Err("No file extension found. Please specify .json, .yaml, or .yml".to_string()),
        }
    }
}

/// Searches a given tree for some given queries and saves results to a file.
///
/// This function is responsible for searching a given tree for some given queries.
/// For each instance, it applies all query algorithms to the tree and saves the results
/// to a single file in the format determined by the file extension.
///
/// # Arguments
/// * `inp_data` - The input data to search.
/// * `tree_path` - The path to the tree to search.
/// * `instances` - The instances to search for.
/// * `query_algorithms` - The query algorithms to apply.
/// * `metric` - The metric to use for searching.
/// * `output_path` - The path to the output file (format determined by extension).
///
/// # Returns
/// A `Result` containing either `Ok(())` if the search was successful, or `Err(String)` if an error occurred.
pub fn search_tree<P: AsRef<Path>, O: AsRef<Path>>(
    inp_data: ShellFlatVec,
    tree_path: P,
    instances: ShellFlatVec,
    query_algorithms: Vec<crate::search::QueryAlgorithm<f64>>,
    metric: ShellMetric,
    output_path: O,
) -> Result<(), String> {
    // Determine output format from file extension
    let format = OutputFormat::from_path(&output_path)?;

    // Create parent directory if it doesn't exist
    if let Some(parent) = output_path.as_ref().parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create parent directory: {e}"))?;
    }

    let tree = ShellTree::read_from(tree_path)?;
    match (&inp_data, &tree, &instances, &metric) {
        (ShellFlatVec::F32(d), ShellTree::Ball(ShellBall::F32(c)), ShellFlatVec::F32(i), ShellMetric::Euclidean(m)) => {
            search(d, m, c, i, &query_algorithms, &output_path, format)
        }
        (ShellFlatVec::F64(d), ShellTree::Ball(ShellBall::F64(c)), ShellFlatVec::F64(i), ShellMetric::Euclidean(m)) => {
            search(d, m, c, i, &query_algorithms, &output_path, format)
        }
        (
            ShellFlatVec::F32(d),
            ShellTree::PermutedBall(ShellPermutedBall::F32(c)),
            ShellFlatVec::F32(i),
            ShellMetric::Euclidean(m),
        ) => search(d, m, c, i, &query_algorithms, &output_path, format),
        (
            ShellFlatVec::F64(d),
            ShellTree::PermutedBall(ShellPermutedBall::F64(c)),
            ShellFlatVec::F64(i),
            ShellMetric::Euclidean(m),
        ) => search(d, m, c, i, &query_algorithms, &output_path, format),
        _ => Err(format!(
            "Unsupported type combination for search:\n\
                 - Input data type: {inp_data}\n\
                 - Tree data type: {tree}\n\
                 - Query data type: {instances}\n\
                 - Metric: {metric}\n\
                 \n\
                 Currently supported combinations:\n\
                 - F32 data + Ball<F32> tree + F32 queries + Euclidean metric\n\
                 - F64 data + Ball<F64> tree + F64 queries + Euclidean metric\n\
                 - F32 data + PermutedBall<F32> tree + F32 queries + Euclidean metric\n\
                 - F64 data + PermutedBall<F64> tree + F64 queries + Euclidean metric\n\
                 \n\
                 Note: The input data (-i flag) should point to your training data file.\n\
                 Example:\n\
                 cargo run --package shell -- -i training.npy cakes search -t ./tree/tree.bin -i queries.npy -q knn-linear:k=5 -o results.json"
        )),
    }?;

    Ok(())
}

fn search<
    I: std::fmt::Debug,
    T: Number + 'static,
    C: Cluster<T>,
    M: Metric<I, T>,
    D: Searchable<I, T, C, M>,
    P: AsRef<Path>,
>(
    data: &D,
    metric: &M,
    root: &C,
    instances: &FlatVec<I, usize>,
    algs: &[crate::search::QueryAlgorithm<f64>],
    output_path: P,
    format: OutputFormat,
) -> Result<(), String> {
    let mut all_results = SearchResults { queries: Vec::new() };

    for (query_index, instance) in instances.items().iter().enumerate() {
        println!("Processing query {query_index}: {instance:?}");

        let mut query_result = QueryResult {
            query_index,
            algorithms: Vec::new(),
        };

        for alg in algs {
            let result = alg.get().search(data, metric, root, instance);
            println!("Result {alg}: {result:?}");

            // Convert result to f64 for serialization consistency
            let neighbors: Vec<(usize, f64)> = result.into_iter().map(|(idx, dist)| (idx, dist.as_f64())).collect();

            query_result.algorithms.push(AlgorithmResult {
                algorithm: alg.to_string(),
                neighbors,
            });
        }

        all_results.queries.push(query_result);
    }

    // Save all results to the specified file
    save_results(&all_results, &output_path, format)?;

    Ok(())
}

/// Saves search results to a file in the specified format.
fn save_results<P: AsRef<Path>>(results: &SearchResults, output_path: P, format: OutputFormat) -> Result<(), String> {
    let output_path = output_path.as_ref();

    let content = match format {
        OutputFormat::Json => {
            serde_json::to_string_pretty(results).map_err(|e| format!("Failed to serialize to JSON: {e}"))?
        }
        OutputFormat::Yaml => serde_yaml::to_string(results).map_err(|e| format!("Failed to serialize to YAML: {e}"))?,
    };

    fs::write(output_path, content).map_err(|e| format!("Failed to write file {}: {}", output_path.display(), e))?;

    println!("Saved search results to {}", output_path.display());

    Ok(())
}
