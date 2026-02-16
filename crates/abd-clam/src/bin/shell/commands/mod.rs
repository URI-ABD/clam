//! Commands for the CLAM shell.

use std::path::PathBuf;

use abd_clam::PartitionStrategy;
use clap::Subcommand;

use crate::{tree::data::ShellDataType, utils::ReportFormat};

use super::tree::ShellMetric;

mod cakes;
mod explore;
mod generate_data;
mod musals;
mod pancakes;

/// The top-level commands for the CLAM shell.
#[derive(Subcommand, Debug)]
#[non_exhaustive]
pub enum Commands {
    /// Build a CLAM tree from a dataset.
    Build {
        /// The type of the input data.
        #[arg(short('t'), long)]
        data_type: ShellDataType,
        /// The number of samples to use for building the tree. If not specified, the entire dataset will be used.
        #[arg(short('n'), long)]
        num_samples: Option<usize>,
        /// The metric to use for building the tree.
        #[arg(short('m'), long)]
        metric: ShellMetric,
        /// The partition strategy to use for building the tree. If not specified, the default strategy will be used.
        #[arg(short('p'), long)]
        partition_strategy: Option<String>,
    },
    /// Entropy-scaling search.
    Cakes {
        /// The format of the output file. If not provided, we will use `json`.
        #[arg(short('f'), long)]
        out_fmt: Option<ReportFormat>,
        /// The path to the queries file.
        #[arg(short('q'), long)]
        queries_path: PathBuf,
        /// If not provided, we will use all the queries in the queries file. If provided, we will use the first `num_queries` queries in the queries file.
        #[arg(short('n'), long)]
        num_queries: Option<usize>,
        /// The algorithm to use for the action.
        #[arg(short('a'), long)]
        algorithm: String,
        /// The specific action to perform with CAKES.
        #[clap(subcommand)]
        action: cakes::Action,
    },
    /// Compressive search built on top of CAKES.
    Pancakes {
        /// The format of the output file. If not provided, we will use `json`.
        #[arg(short('f'), long)]
        out_fmt: Option<ReportFormat>,
        /// The specific action to perform with Pancakes.
        #[clap(subcommand)]
        action: pancakes::Action,
    },
    /// Multiple Sequence Alignment
    Musals {
        /// The format of the output file. If not provided, we will use `json`.
        #[arg(short('f'), long)]
        out_fmt: Option<ReportFormat>,
        /// The specific action to perform with MUSALS.
        #[clap(subcommand)]
        action: musals::Action,
    },
    /// Generate synthetic vector datasets for testing and benchmarking
    GenerateData {
        /// The number of items to generate.
        #[arg(short('n'), long)]
        num_items: usize,
        /// The size of each item (e.g., the length of each sequence or dimensionality of each vector).
        #[arg(short('s'), long)]
        item_size: usize,
    },
    /// Explore various properties of datasets and algorithms
    Explore {
        /// The specific aspect to explore.
        #[clap(subcommand)]
        action: explore::Action,
    },
}

impl Commands {
    /// Execute the command.
    pub fn execute<P: AsRef<std::path::Path>, R: rand::Rng>(
        &self,
        inp_path: &P,
        out_dir: &P,
        rng: &mut R,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match self {
            Self::Build {
                data_type,
                num_samples,
                metric,
                partition_strategy,
            } => {
                // Check that the input path exists and is a file.
                let data_path = crate::utils::check_path_exists(inp_path, false, false)?;

                // Parse the partition strategy if provided.
                let partition_strategy = if let Some(partition_strategy) = partition_strategy {
                    partition_strategy.parse()?
                } else {
                    PartitionStrategy::default()
                };

                // Infer the name of the output tree file based on the inputs, and create an output path for it.
                let tree_name = crate::tree::infer_tree_name(&data_path, data_type, metric, &partition_strategy)?;
                let tree_out_dir = out_dir.as_ref().join("trees");
                crate::utils::check_path_exists(&tree_out_dir, true, true)?;
                let out_path = tree_out_dir.join(&tree_name);
                if out_path.exists() {
                    ftlog::warn!("Output tree file already exists: {out_path:?}. It will be overwritten.");
                }

                // Build the tree and write it to the output path.
                ftlog::info!("Building tree from: {data_path:?} with metric: {metric} and partition strategy: {partition_strategy}...");
                let tree = crate::tree::ShellTree::build(&data_path, data_type, rng, *num_samples, metric, &partition_strategy)?;
                ftlog::info!("Tree built successfully.");
                tree.write(&out_path)?;
                ftlog::info!("Tree written to: {out_path:?}");
            }
            Self::Cakes {
                out_fmt,
                queries_path,
                num_queries,
                algorithm,
                action,
            } => {
                // Check that the input path exists and is a file.
                let tree_path = crate::utils::check_path_exists(inp_path, false, false)?;

                // Perform the specified action on the tree, and write the results in the specified output format.
                let out_fmt = out_fmt.unwrap_or_default();
                action.perform(&tree_path, out_dir.as_ref(), out_fmt, queries_path, *num_queries, algorithm, rng)?;
            }
            Self::Pancakes { .. } => todo!("Pancakes command not implemented yet."),
            Self::Musals { out_fmt, action } => {
                // Check that the input path exists and is a file.
                let tree_path = crate::utils::check_path_exists(inp_path, false, false)?;

                // Perform the specified action on the tree, and write the results in the specified output format.
                let out_fmt = out_fmt.unwrap_or_default();
                action.perform(&tree_path, out_dir.as_ref(), out_fmt, rng)?;
            }
            Self::GenerateData { .. } => todo!("GenerateData command not implemented yet."),
            Self::Explore { .. } => todo!("Explore command not implemented yet."),
        }
        Ok(())
    }
}
