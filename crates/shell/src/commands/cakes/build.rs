//! Building the tree for search

use std::path::Path;

use crate::{data::ShellData, metrics::Metric, trees::ShellTree};

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
/// - `out_dir`: The output directory to write the tree and data to.
///
/// # Errors
///
/// - If the dataset and metric are deemed an incompatible combination. See
///   [`ShellTree::new`](crate::trees::ShellTree::new) for more details.
pub fn build_new_tree<P: AsRef<Path>>(inp_data: ShellData, metric: &Metric, out_dir: P) -> Result<(), String> {
    let tree = ShellTree::new(inp_data, metric)?;
    let tree_path = tree.tree_file_path(&out_dir, None);
    tree.write_to(&tree_path)
}
