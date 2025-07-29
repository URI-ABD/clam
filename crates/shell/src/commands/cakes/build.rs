//! Building the tree for search

use std::path::Path;

use crate::{data::ShellFlatVec, metrics::ShellMetric, trees::ShellTree};

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
