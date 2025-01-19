//! Datasets for use in MSA experiments.

use std::path::Path;

use abd_clam::{dataset::AssociatesMetadata, msa::MSA};

mod raw;

use distances::Number;
pub use raw::FastaFile;

/// Write a CLAM `Dataset` to a FASTA file.
pub fn write_fasta<P: AsRef<Path>, T: Number>(data: &MSA<String, T, String>, path: P) -> Result<(), String> {
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file {path:?}: {e}"))?;
    let mut writer = bio::io::fasta::Writer::new(file);

    let metadata = data.metadata();
    let sequences = data.data().items();

    for (id, seq) in metadata.iter().zip(sequences.iter()) {
        let record = bio::io::fasta::Record::with_attrs(id, None, seq.as_str().as_bytes());
        writer
            .write_record(&record)
            .map_err(|e| format!("Failed to write record: {e}"))?;
    }

    Ok(())
}
