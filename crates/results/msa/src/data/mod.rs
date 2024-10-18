//! Datasets for use in MSA experiments.

use std::path::Path;

use abd_clam::FlatVec;
use distances::Number;

mod raw;

#[allow(clippy::module_name_repetitions)]
pub use raw::RawData;

/// Write a CLAM `Dataset` to a FASTA file.
pub fn write_fasta<U: Number, P: AsRef<Path>>(data: &FlatVec<String, U, String>, path: P) -> Result<(), String> {
    let path = path.as_ref();
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file {path:?}: {e}"))?;
    let mut writer = bio::io::fasta::Writer::new(file);

    let metadata = data.metadata();
    let sequences = data.instances();

    for (id, seq) in metadata.iter().zip(sequences.iter()) {
        let record = bio::io::fasta::Record::with_attrs(id, None, seq.as_bytes());
        writer
            .write_record(&record)
            .map_err(|e| format!("Failed to write record: {e}"))?;
    }

    Ok(())
}
