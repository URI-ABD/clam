//! Module for loading datasets.

use abd_clam::FlatVec;

mod h5;
mod npy;

/// Reads a dataset for dimensionality reduction.
pub fn read<P: AsRef<std::path::Path>>(inp_dir: &P, name: &str) -> Result<FlatVec<Vec<f32>, usize>, String> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.len() < 2 {
        return Err(format!("Invalid file name: {name}"));
    }
    let (name, ext) = (parts[..parts.len() - 1].join("."), parts[parts.len() - 1]);
    match ext {
        "npy" => npy::read(&inp_dir.as_ref().join(format!("{name}.npy"))),
        "h5" | "hdf5" => h5::read(inp_dir, &name),
        _ => Err(format!("Unsupported file extension: {ext}")),
    }
}
