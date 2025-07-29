//! Search a given tree for some given queries.

use std::path::Path;

use crate::{data::ShellFlatVec, metrics::ShellMetric, trees::ShellTree};

#[allow(dead_code, unused_variables)]
pub fn search_tree<P: AsRef<Path>>(
    tree_path: P,
    data_path: P,
    queries_path: P,
    out_path: P,
    metric: ShellMetric,
) -> Result<(), String> {
    // Load the tree and data
    let tree = ShellTree::read_from(tree_path)?;
    let data = ShellFlatVec::read_from(data_path)?;

    todo!("Tom")
}
