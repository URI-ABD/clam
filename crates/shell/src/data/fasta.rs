use std::path::Path;

/// Reads a FASTA file from the given path.
#[allow(dead_code, unused_variables)]
pub fn read<P: AsRef<Path>>(path: P) -> Result<Vec<(String, String)>, String> {
    todo!("Najib: Implement reading FASTA files")
}
