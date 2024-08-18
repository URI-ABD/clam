//! Readers for `GreenGenes` datasets.

/// Reads a `GreenGenes` dataset from the given path.
pub fn read(path: &std::path::Path, num_seqs: usize) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    println!("Reading GreenGenes dataset from {path:?}. Will read {num_seqs} sequence(s).");

    Ok(())
}
