//! Data formats supported in the CLI.

mod fasta;
mod npy;

use std::path::Path;

use abd_clam::FlatVec;

/// Reads the data from the file at the given path.
pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellFlatVec, String> {
    match Format::from(&path) {
        Format::Npy => ShellFlatVec::read_npy(path),
        Format::Fasta => ShellFlatVec::read_fasta(path),
    }
}

/// Data formats supported in the CLI.
pub enum Format {
    /// Npy array format.
    Npy,
    /// FASTA format.
    Fasta,
}

impl<P: AsRef<Path>> From<P> for Format {
    fn from(path: P) -> Self {
        match path.as_ref().extension().and_then(|s| s.to_str()) {
            Some("npy") => Format::Npy,
            Some("fasta") => Format::Fasta,
            Some(ext) => panic!("Unknown data format {ext} for path: {}", path.as_ref().display()),
            None => panic!(
                "Could not determine data format without extension for path: {}",
                path.as_ref().display()
            ),
        }
    }
}

#[derive(bitcode::Encode, bitcode::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellFlatVec {
    /// FlatVec for strings from FASTA files.
    String(FlatVec<String, usize>),
    /// FlatVec for various numeric types from NPY files.
    F32(FlatVec<Vec<f32>, usize>),
    F64(FlatVec<Vec<f64>, usize>),
    I8(FlatVec<Vec<i8>, usize>),
    I16(FlatVec<Vec<i16>, usize>),
    I32(FlatVec<Vec<i32>, usize>),
    I64(FlatVec<Vec<i64>, usize>),
    U8(FlatVec<Vec<u8>, usize>),
    U16(FlatVec<Vec<u16>, usize>),
    U32(FlatVec<Vec<u32>, usize>),
    U64(FlatVec<Vec<u64>, usize>),
}

impl ShellFlatVec {
    /// Reads a NPY file and returns a ShellFlatVec.
    pub fn read_npy<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        npy::NpyType::read(path)
    }

    /// Reads a FASTA file and returns a ShellFlatVec.
    pub fn read_fasta<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        fasta::read(path)
    }

    /// Saves the ShellFlatVec to the specified path using bincode.
    pub fn write_to<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let contents = bitcode::encode(self).map_err(|e| e.to_string())?;
        std::fs::write(path, contents).map_err(|e| e.to_string())
    }

    /// Reads a ShellFlatVec from the specified path using bincode.
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let contents = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&contents).map_err(|e| e.to_string())
    }
}
