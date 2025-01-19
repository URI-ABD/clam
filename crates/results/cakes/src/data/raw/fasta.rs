//! Reading FASTA files

use abd_clam::{dataset::AssociatesMetadata, Dataset};
use clap::error::Result;

/// Reads a FASTA file from the given path.
///
/// # Arguments
///
/// * `path`: The path to the FASTA file.
/// * `holdout`: The number of sequences to hold out for queries.
/// * `remove_gaps`: Whether to remove gaps from the sequences.
///
/// # Returns
///
/// * The sequences and queries.
/// # The minimum and maximum lengths of the sequences.
///
/// # Errors
///
/// * If the file does not exist.
/// * If the extension is not `.fasta`.
/// * If the file cannot be read as a FASTA file.
/// * If any ID or sequence is empty.
#[allow(clippy::type_complexity)]
pub fn read<P: AsRef<std::path::Path>>(
    path: &P,
    holdout: usize,
    remove_gaps: bool,
) -> Result<([Vec<(String, String)>; 2], [usize; 2]), String> {
    let (data, queries) = bench_utils::fasta::read(path, holdout, remove_gaps)?;
    let (min_len, max_len) = data.dimensionality_hint();
    let max_len = max_len.unwrap_or(min_len);

    let seqs = data.items();
    let ids = data.metadata();

    let seqs = ids.iter().cloned().zip(seqs.iter().cloned()).collect();

    #[allow(clippy::tuple_array_conversions)]
    Ok(([seqs, queries], [min_len, max_len]))
}
