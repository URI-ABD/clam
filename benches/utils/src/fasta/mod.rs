//! Utilities for dealing with FASTA files.

use std::path::Path;

use abd_clam::{
    dataset::{AssociatesMetadata, AssociatesMetadataMut},
    Dataset, FlatVec,
};
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
/// * A `Vec` of the queries. Each query is a tuple of the ID and sequence.
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
    let name = path
        .file_stem()
        .ok_or("No file name found")?
        .to_string_lossy()
        .to_string();
    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    if path.extension().is_none_or(|ext| ext != "fasta") {
        return Err(format!("Path {path:?} does not have the `.fasta` extension!"));
    }

    ftlog::info!("Reading FASTA file from {path:?}.");

    let mut records = bio::io::fasta::Reader::from_file(path)
        .map_err(|e| e.to_string())?
        .records();

    // Create accumulator for sequences and track the min and max lengths.
    let mut seqs = Vec::new();
    let mut lengths = Vec::new();

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
                .filter(|&c| *c != b'-' && *c != b'.')
                .map(|&c| c as char)
                .collect()
        } else {
            record.seq().iter().map(|&c| c as char).collect()
        };

        if seq.is_empty() {
            return Err(format!("Empty sequence for record {}.", seqs.len()));
        }

        lengths.push(seq.len());

        // Add the sequence to the accumulator.
        seqs.push((name, seq));

        if seqs.len() % 10_000 == 0 {
            // Log progress every 10,000 sequences.
            ftlog::debug!("Read {} sequences...", seqs.len());
        }
    }

    if seqs.is_empty() {
        return Err("No sequences found!".to_string());
    }
    ftlog::info!("Read {} sequences.", seqs.len());

    // Shuffle the sequences and split off the queries.
    let queries = if holdout > 0 {
        let mut rng = rand::rng();
        seqs.shuffle(&mut rng);
        seqs.split_off(seqs.len() - holdout)
    } else {
        Vec::new()
    };
    ftlog::info!("Holding out {} queries.", queries.len());

    // Compute statistics for the lengths of the sequences.
    let (min_len, max_len, _, _, _) = len_stats(&lengths);

    // Unzip the IDs and sequences.
    let (ids, seqs): (Vec<_>, Vec<_>) = seqs.into_iter().unzip();

    // Create the FlatVec.
    let data = FlatVec::new(seqs)?
        .with_dim_lower_bound(min_len)
        .with_dim_upper_bound(max_len)
        .with_metadata(&ids)?
        .with_name(&name);

    Ok((data, queries))
}

/// Computes statistics for the lengths of sequences and logs them.
///
/// # Arguments
///
/// * `lengths`: The lengths of the sequences.
///
/// # Returns
///
/// * The minimum length.
/// * The maximum length.
/// * The median length.
/// * The mean length.
/// * The standard deviation of the lengths.
#[must_use]
pub fn len_stats(lengths: &[usize]) -> (usize, usize, usize, f32, f32) {
    let lengths = {
        let mut lengths = lengths.to_vec();
        lengths.sort_unstable();
        lengths
    };
    let (min_len, max_len) = lengths.iter().fold((usize::MAX, 0), |(min, max), &len| {
        (Ord::min(min, len), Ord::max(max, len))
    });
    let median_len = lengths[lengths.len() / 2];
    let mean_len: f32 = abd_clam::utils::mean(&lengths);
    let std_len: f32 = abd_clam::utils::standard_deviation(&lengths);
    ftlog::info!("Length stats: min = {min_len}, max = {max_len}, median = {median_len}, mean = {mean_len:.2}, std = {std_len:.2}.");
    (min_len, max_len, median_len, mean_len, std_len)
}

/// Writes a dataset to a FASTA file at the given path.
///
/// # Arguments
///
/// * `data`: The data to write.
/// * `path`: The path to write the FASTA file to.
///
/// # Errors
///
/// * If the file cannot be written.
/// * If any ID or sequence is empty.
pub fn write<S, D, P>(data: &D, path: &P) -> Result<(), String>
where
    S: AsRef<str>,
    D: AssociatesMetadata<S, String>,
    P: AsRef<Path>,
{
    let path = path.as_ref();
    ftlog::info!("Writing FASTA file to {path:?}.");

    let mut writer = bio::io::fasta::Writer::to_file(path).map_err(|e| e.to_string())?;

    let metadata = data.metadata().iter();
    let items = (0..data.cardinality()).map(|i| data.get(i));

    for (id, seq) in metadata.zip(items) {
        writer
            .write_record(&bio::io::fasta::Record::with_attrs(
                id.as_ref(),
                None,
                seq.as_ref().as_bytes(),
            ))
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}
