//! TODO: Emily

use std::path::Path;

use abd_clam::{Dataset, FlatVec};

use super::ShellFlatVec;

/// Reads a FASTA file from the given path.
#[allow(dead_code, unused_variables)]
pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellFlatVec, String> {
    // See the `read` function in `benches/utils/src/fasta/mod.rs` as a reference and guide for implementation.
    todo!("Emily")
}

/// Writes a FASTA file to the given path.
pub fn write<P: AsRef<Path>>(path: P, data: &FlatVec<String, usize>) -> Result<(), String> {
    use std::io::Write;

    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create FASTA file: {e}"))?;
    let mut writer = std::io::BufWriter::new(file);

    // Write each sequence with a simple numeric ID
    for i in 0..data.cardinality() {
        let sequence = data.get(i);
        writeln!(writer, ">sequence_{i}").map_err(|e| format!("Failed to write sequence header: {e}"))?;
        writeln!(writer, "{sequence}").map_err(|e| format!("Failed to write sequence data: {e}"))?;
    }

    writer.flush().map_err(|e| format!("Failed to flush FASTA file: {e}"))?;

    Ok(())
}
