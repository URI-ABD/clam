//! Commands for the CAKES algorithm.

use std::collections::HashSet;

use abd_clam::cakes::MeasurableSearchQuality;
use clap::Subcommand;

use crate::{tree::ShellTree, utils::ReportFormat};

/// The specific action to perform with CAKES.
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Action {
    /// Perform a search using the provided CAKES algorithm.
    Search,
    /// Benchmark the requested CAKES algorithms on the provided queries to measure throughput, and the additional quality metrics if requested.
    Bench {
        /// The quality metrics to compute. If not provided, we will compute all available quality metrics.
        #[arg(short('c'), long)]
        quality_metrics: Option<Vec<MeasurableSearchQuality>>,
    },
    /// Explore the throughput vs [quality-metric] tradeoff frontier for the requested CAKES algorithms on the provided queries using the given range of `tol`
    /// values. We will adaptively select `tol` values to explore the shape of the frontier.
    Frontier {
        /// The quality metrics to compute. If not provided, we will compute all available quality metrics.
        #[arg(short('c'), long)]
        quality_metrics: Option<Vec<MeasurableSearchQuality>>,
        /// The range of `tol` values to explore (e.g., "0.01,0.99" for the default range of [0.01, 0.99]).
        #[arg(short('t'), long, value_parser = parse_tol_range)]
        tol_range: Option<(f64, f64)>,
        /// The minimum delta between `tol` values to consider when exploring the frontier. If not provided, we will use a default value of 0.01.
        #[arg(short('d'), long)]
        tol_delta: Option<f64>,
    },
}

/// Parse a `tol` range from a string in the format `tol_min,tol_max`.
fn parse_tol_range(s: &str) -> Result<(f64, f64), Box<dyn std::error::Error + Send + Sync>> {
    let re = lazy_regex::regex!(r"^([0-9]*\.?[0-9]+),([0-9]*\.?[0-9]+)$");
    let caps = re.captures(s).ok_or_else(|| format!("Invalid tol range format: {s}"))?;
    let tol_min: f64 = caps[1].parse()?;
    let tol_max: f64 = caps[2].parse()?;
    if !(0.0..=1.0).contains(&tol_min) || !(0.0..=1.0).contains(&tol_max) {
        return Err(format!("Tol values must be in the range [0, 1]: {s}").into());
    }
    if tol_min >= tol_max {
        return Err(format!("Tol min must be less than tol max: {s}").into());
    }
    Ok((tol_min, tol_max))
}

impl Action {
    /// Perform the specified action for the CAKES command.
    ///
    /// # Arguments
    ///
    /// - `tree`: The tree to perform the action on.
    /// - `out_dir`: The output directory to write any results to.
    /// - `out_fmt`: The output format to use when writing results.
    /// - `rng`: The random number generator to use for any random operations.
    #[expect(unused, clippy::needless_pass_by_ref_mut, clippy::too_many_arguments)]
    pub fn perform<P: AsRef<std::path::Path>, R: rand::Rng>(
        &self,
        tree_path: &P,
        out_dir: &std::path::Path,
        out_fmt: ReportFormat,
        queries_path: &P,
        num_queries: Option<usize>,
        algorithm: &str,
        rng: &mut R,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Search => {
                let out_name = format!("cakes-search-hits-{}.{}", algorithm, out_fmt.file_extension());
                let out_path = out_dir.join(out_name);
                if out_path.exists() {
                    ftlog::warn!("Output file already exists: {out_path:?}. It will be overwritten.");
                }

                ftlog::info!("Reading tree from {:?}...", tree_path.as_ref());
                let tree = ShellTree::read(tree_path)?;

                ftlog::info!("Performing search with algorithm {algorithm} on tree from {:?}...", tree_path.as_ref());
                let hits = tree.search(algorithm, queries_path, num_queries)?;

                ftlog::info!("Writing search results to {out_path:?}...");
                out_fmt.write_report(&hits, &out_path, true)
            }
            Self::Bench { quality_metrics } => {
                let quality_metrics = quality_metrics.as_ref().map_or_else(
                    || MeasurableSearchQuality::all_variants().into_iter().collect(),
                    |quality_metrics| quality_metrics.iter().copied().collect::<HashSet<_>>(),
                );
                ftlog::info!("Will compute {} quality metrics for the benchmark:", quality_metrics.len());
                for qm in &quality_metrics {
                    ftlog::info!("- {qm:?}");
                }

                let out_name = format!("cakes-bench-{}.{}", algorithm, out_fmt.file_extension());
                let out_path = out_dir.join(out_name);
                if out_path.exists() {
                    ftlog::warn!("Output file already exists: {out_path:?}. It will be overwritten.");
                }

                ftlog::info!("Reading tree from {:?}...", tree_path.as_ref());
                let tree = ShellTree::read(tree_path)?;

                ftlog::info!(
                    "Performing benchmark with algorithm {algorithm} on tree from {:?} with quality metrics {}...",
                    tree_path.as_ref(),
                    quality_metrics.len()
                );
                let benchmark = tree.bench(algorithm, queries_path, num_queries, quality_metrics)?;

                ftlog::info!("Summarizing benchmark results...");
                let summary = benchmark.summarize();

                ftlog::info!("Writing benchmark results to {out_path:?}...");
                out_fmt.write_report(&summary, &out_path, true)
            }
            Self::Frontier {
                quality_metrics,
                tol_range,
                tol_delta,
            } => {
                let quality_metrics = quality_metrics.as_ref().map_or_else(
                    || MeasurableSearchQuality::all_variants().into_iter().collect(),
                    |quality_metrics| quality_metrics.iter().copied().collect::<HashSet<_>>(),
                );
                ftlog::info!("Quality metrics for frontier exploration: {quality_metrics:?}");

                let (min_tol, max_tol) = tol_range.unwrap_or((0.01, 0.99));
                let tol_delta = tol_delta.unwrap_or(0.01);

                todo!(
                    "Perform CAKES Frontier with queries_path={:?}, num_queries={num_queries:?}, algorithm={algorithm}, quality_metrics={quality_metrics:?}, tol_range=({min_tol}, {max_tol}), tol_delta={tol_delta}",
                    queries_path.as_ref()
                );
            }
        }
    }
}
