//! Building the tree for search

use std::path::Path;

use crate::{data::ShellFlatVec, metrics::ShellMetric, trees::ShellTree};

/// Builds and writes the tree and data to the specified output directory.
///
/// This function is responsible for creating a new tree for the input data,
/// making any permutations or transformations as needed on the data, and
/// finally writing the tree and data in the specified output directory. The
/// tree and data will be written to separate files named `tree.bin` and
/// `data.bin` respectively.
///
/// # Arguments
///
/// - `inp_data`: The input data to build the tree from.
/// - `metric`: The distance metric to use for the tree.
/// - `seed`: The random seed to use.
/// - `balanced`: Whether to build a balanced tree.
/// - `permuted`: Whether to apply depth-first-reordering to the data.
/// - `out_dir`: The output directory to write the tree and data to.
///
/// # Errors
///
/// - If the dataset and metric are deemed an incompatible combination. See
///   [`ShellTree::new`](crate::trees::ShellTree::new) for more details.
pub fn build_new_tree<P: AsRef<Path>>(
    inp_data: ShellFlatVec,
    metric: ShellMetric,
    seed: Option<u64>,
    balanced: bool,
    permuted: bool,
    out_dir: P,
) -> Result<(), String> {
    let (ball, data) = ShellTree::new(inp_data, &metric, seed, balanced, permuted)?;

    let tree_path = out_dir.as_ref().join("tree.bin");
    ball.write_to(tree_path)?;

    let data_path = out_dir.as_ref().join("data.bin");
    data.write_to(data_path)?;

    Ok(())
}
