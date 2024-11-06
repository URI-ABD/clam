//! Reading FASTA files

use std::path::Path;

use clap::error::Result;
use rand::seq::SliceRandom;

/// A collection of named sequences.
type NamedSequences = Vec<(String, String)>;

/// Reads a FASTA file from the given path.
///
/// # Arguments
///
/// * `path`: The path to the FASTA file.
/// * `holdout`: The number of sequences to hold out for queries.
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
pub fn read<P: AsRef<Path>>(path: P, holdout: usize) -> Result<([NamedSequences; 2], [usize; 2]), String> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    if !path.extension().map_or(false, |ext| ext == "fasta") {
        return Err(format!("Path {path:?} does not have the `.fasta` extension!"));
    }

    ftlog::info!("Reading FASTA file from {path:?}.");

    let mut records = bio::io::fasta::Reader::from_file(path)
        .map_err(|e| e.to_string())?
        .records();

    let mut seqs = Vec::new();
    let (mut min_len, mut max_len) = (usize::MAX, 0);

    while let Some(Ok(record)) = records.next() {
        let name = record.id().to_string();
        if name.is_empty() {
            return Err(format!("Empty ID for record {}.", seqs.len()));
        }

        let seq = record.seq().iter().map(|&b| b as char).collect::<String>();
        if seq.is_empty() {
            return Err(format!("Empty sequence for record {} with ID {name}.", seqs.len()));
        }

        min_len = min_len.min(seq.len());
        max_len = max_len.max(seq.len());

        seqs.push((name, seq));
    }

    ftlog::info!("Read {} sequences from {path:?}.", seqs.len());
    ftlog::info!("Minimum length: {min_len}, Maximum length: {max_len}.");

    // Shuffle the sequences and hold out a query set.
    seqs.shuffle(&mut rand::thread_rng());
    let queries = seqs.split_off(seqs.len() - holdout);

    #[allow(clippy::tuple_array_conversions)]
    Ok(([seqs, queries], [min_len, max_len]))
}
