//! Building the tree for search

use std::path::Path;

use crate::{data, metrics::ShellMetric, trees::ShellTree};

pub fn build_new_tree<P: AsRef<Path>>(
    inp_path: P,
    out_path: P,
    tree_path: P,
    metric: ShellMetric,
    seed: Option<u64>,
    balanced: bool,
    permuted: bool,
) -> Result<(), String> {
    let (ball, data) = ShellTree::new(data::Format::read(inp_path)?, &metric, seed, balanced, permuted)?;

    ball.write_to(tree_path)?;
    data.write_to(out_path)?;

    Ok(())
}
