//! Utilities for dealing with FASTA files.

use std::path::Path;

use abd_clam::{dataset::AssociatesMetadataMut, FlatVec};
use rand::prelude::*;

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
/// * The training sequences in a `FlatVec`.
/// * A `Vec` of the queries.
///
/// # Errors
///
/// * If the file does not exist.
/// * If the extension is not `.fasta`.
/// * If the file cannot be read as a FASTA file.
/// * If any ID or sequence is empty.
#[allow(clippy::type_complexity)]
pub fn read<P: AsRef<Path>>(
    path: &P,
    holdout: usize,
    remove_gaps: bool,
) -> Result<(FlatVec<String, String>, Vec<(String, String)>), String> {
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

    // Create accumulator for sequences and track the min and max lengths.
    let mut seqs = Vec::new();
    let (mut min_len, mut max_len) = (usize::MAX, 0);

    // Check each record for an empty ID or sequence.
    while let Some(Ok(record)) = records.next() {
        let name = record.id().to_string();
        if name.is_empty() {
            return Err(format!("Empty ID for record {}.", seqs.len()));
        }

        let seq: String = if remove_gaps {
            record
                .seq()
                .iter()
                .filter(|&c| *c != b'-')
                .map(|&c| c as char)
                .collect()
        } else {
            record.seq().iter().map(|&c| c as char).collect()
        };

        if seq.is_empty() {
            return Err(format!("Empty sequence for record {}.", seqs.len()));
        }

        if seq.len() < min_len {
            min_len = seq.len();
        }
        if seq.len() > max_len {
            max_len = seq.len();
        }

        // Add the sequence to the accumulator.
        seqs.push((name, seq));

        if seqs.len() % 10_000 == 0 {
            // Log progress every 10,000 sequences.
            ftlog::info!("Read {} sequences...", seqs.len());
        }
    }

    if seqs.is_empty() {
        return Err("No sequences found!".to_string());
    }
    ftlog::info!("Read {} sequences.", seqs.len());

    // Shuffle the sequences and split off the queries.
    let queries = if holdout > 0 {
        let mut rng = rand::thread_rng();
        seqs.shuffle(&mut rng);
        seqs.split_off(seqs.len() - holdout)
    } else {
        Vec::new()
    };
    ftlog::info!("Holding out {} queries.", queries.len());

    // Unzip the IDs and sequences.
    let (ids, seqs): (Vec<_>, Vec<_>) = seqs.into_iter().unzip();

    // Create the FlatVec.
    let data = FlatVec::new(seqs)?
        .with_dim_lower_bound(min_len)
        .with_dim_upper_bound(max_len)
        .with_metadata(&ids)?;

    Ok((data, queries))
}
