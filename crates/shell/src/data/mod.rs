//! Data formats supported in the CLI.

use std::path::Path;

mod fasta;
pub mod npy;

/// Reads the data from the file at the given path.
pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellData, String> {
    match Format::from(&path) {
        Format::Npy => ShellData::read_npy(path),
        Format::Fasta => ShellData::read_fasta(path),
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
pub enum ShellData {
    /// Vec of sequences and their metadata from FASTA files.
    String(Vec<(String, String)>),
    /// Vec of various numeric types from NPY files.
    F32(Vec<Vec<f32>>),
    F64(Vec<Vec<f64>>),
    I8(Vec<Vec<i8>>),
    I16(Vec<Vec<i16>>),
    I32(Vec<Vec<i32>>),
    I64(Vec<Vec<i64>>),
    U8(Vec<Vec<u8>>),
    U16(Vec<Vec<u16>>),
    U32(Vec<Vec<u32>>),
    U64(Vec<Vec<u64>>),
}

impl ShellData {
    /// Reads a NPY file and returns a ShellFlatVec.
    pub fn read_npy<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        npy::NpyType::read(path)
    }

    /// Reads a FASTA file and returns a ShellFlatVec.
    pub fn read_fasta<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        fasta::read(path).map(ShellData::String)
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
