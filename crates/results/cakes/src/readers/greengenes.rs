//! Readers for `GreenGenes` datasets.

use abd_clam::{FlatVec, Metric};
use bio::io::fasta;

use super::FlatGenomic;

/// Reads a `GreenGenes` dataset from the given path.
pub fn read(path: &std::path::Path) -> Result<FlatGenomic, String> {
    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    mt_logger::mt_log!(mt_logger::Level::Info, "Reading GreenGenes dataset from {path:?}.");

    let mut records = fasta::Reader::from_file(path).map_err(|e| e.to_string())?.records();
    let mut num_reads = 0;

    let mut ids = Vec::new();
    let mut seqs = Vec::new();

    while let Some(Ok(record)) = records.next() {
        num_reads += 1;

        let id = record.id().to_string();
        if id.is_empty() {
            return Err(format!("Empty id for record {num_reads}."));
        }

        let seq = record.seq().iter().map(|&b| b as char).collect::<String>();
        if seq.is_empty() {
            return Err(format!("Empty sequence for record {num_reads}."));
        }

        ids.push(id);
        seqs.push(seq);
    }

    let (min_seq_len, max_seq_len) = seqs.iter().fold((usize::MAX, 0), |(min, max), seq| {
        let len = seq.len();
        (min.min(len), max.max(len))
    });

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Read {} sequences from {path:?}. Minimum len: {min_seq_len}, Maximum len: {max_seq_len}.",
        seqs.len()
    );

    let metric = Metric::default();
    Ok(FlatVec::new(seqs, metric)?
        .with_metadata(ids)?
        .with_dim_lower_bound(min_seq_len)
        .with_dim_upper_bound(max_seq_len))
}
