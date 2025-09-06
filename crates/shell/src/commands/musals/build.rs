//! Builds and writes the MSA tree for MuSALS

use std::path::Path;

use crate::{commands::musals::ShellCostMatrix, data::ShellData, metrics::Metric, trees::ShellTree};

/// Builds and writes the MSA tree for MuSALS.
pub fn build_msa<P: AsRef<Path>>(
    inp_data: ShellData,
    metric: &Metric,
    cost_matrix: &ShellCostMatrix,
    out_dir: P,
    save_fasta: bool,
    rebuild: bool,
) -> Result<(), String> {
    let out_dir = out_dir.as_ref();
    let parent = out_dir.parent().ok_or_else(|| format!("Output path '{out_dir:?}' has no parent directory"))?;
    if !parent.exists() {
        return Err(format!("Output directory '{parent:?}' does not exist"));
    }
    if !out_dir.exists() {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("Failed to create output directory '{out_dir:?}': {e}"))?;
    }

    let tree_path = crate::utils::tree_file_path(out_dir, &inp_data, metric, None, Some("unaligned"));
    let msa_tree_path = crate::utils::tree_file_path(out_dir, &inp_data, metric, Some(&format!("msa-{cost_matrix}")), None);

    let tree = if tree_path.exists() && !rebuild {
        ftlog::info!("Loading existing unaligned tree from '{tree_path:?}'...");
        ShellTree::read_from(&tree_path, metric)?
    } else {
        ftlog::info!("Building unaligned tree...");
        let tree = ShellTree::new(inp_data, metric)?;

        ftlog::info!("Writing unaligned tree to '{tree_path:?}'...");
        tree.write_to(&tree_path)?;

        tree
    };

    let msa_tree = match tree {
        ShellTree::Lcs(tree) => ShellTree::Lcs(tree.par_into_msa(&cost_matrix.get())),
        ShellTree::Levenshtein(tree) => ShellTree::Levenshtein(tree.par_into_msa(&cost_matrix.get())),
        _ => return Err("MSA tree can only be built for string data.".to_string()),
    };

    ftlog::info!("Writing MSA tree to '{msa_tree_path:?}'...");
    msa_tree.write_to(&msa_tree_path)?;

    if save_fasta {
        let items = match msa_tree {
            ShellTree::Levenshtein(tree) => tree.take_items(),
            _ => unreachable!(),
        };
        let data = ShellData::String(items);

        let fasta_path = out_dir.join(format!("msa-{cost_matrix}.fasta"));
        ftlog::info!("Saving MSA sequences to '{fasta_path:?}'...");

        data.write(fasta_path)?;
    }

    Ok(())
}
