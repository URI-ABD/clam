//! Data formats supported in the CLI.

mod fasta;
mod npy;

use std::path::Path;
use std::str::FromStr;

use abd_clam::FlatVec;

/// Reads the data from the file at the given path.
pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellFlatVec, String> {
    match Format::from(&path) {
        Format::Npy => ShellFlatVec::read_npy(path),
        Format::Fasta => ShellFlatVec::read_fasta(path),
    }
}

/// Data formats supported in the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Format {
    /// Npy array format.
    Npy,
    /// FASTA format.
    Fasta,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Npy => write!(f, "npy"),
            Format::Fasta => write!(f, "fasta"),
        }
    }
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

impl FromStr for Format {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "npy" => Ok(Format::Npy),
            "fasta" => Ok(Format::Fasta),
            _ => Err(format!("Unknown format: '{}'. Use 'npy' or 'fasta'.", s)),
        }
    }
}

/// Data types supported for generated datasets.
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// String sequences
    String,
}

impl FromStr for DataType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" => Ok(DataType::F32),
            "f64" => Ok(DataType::F64),
            "i8" => Ok(DataType::I8),
            "i16" => Ok(DataType::I16),
            "i32" => Ok(DataType::I32),
            "i64" => Ok(DataType::I64),
            "u8" => Ok(DataType::U8),
            "u16" => Ok(DataType::U16),
            "u32" => Ok(DataType::U32),
            "u64" => Ok(DataType::U64),
            "string" => Ok(DataType::String),
            _ => Err(format!("Unknown data type: {s}")),
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

impl std::fmt::Display for ShellFlatVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellFlatVec::String(vec) => write!(f, "String({})", vec.items().len()),
            ShellFlatVec::F32(vec) => write!(f, "F32({})", vec.items().len()),
            ShellFlatVec::F64(vec) => write!(f, "F64({})", vec.items().len()),
            ShellFlatVec::I8(vec) => write!(f, "I8({})", vec.items().len()),
            ShellFlatVec::I16(vec) => write!(f, "I16({})", vec.items().len()),
            ShellFlatVec::I32(vec) => write!(f, "I32({})", vec.items().len()),
            ShellFlatVec::I64(vec) => write!(f, "I64({})", vec.items().len()),
            ShellFlatVec::U8(vec) => write!(f, "U8({})", vec.items().len()),
            ShellFlatVec::U16(vec) => write!(f, "U16({})", vec.items().len()),
            ShellFlatVec::U32(vec) => write!(f, "U32({})", vec.items().len()),
            ShellFlatVec::U64(vec) => write!(f, "U64({})", vec.items().len()),
        }
    }
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
    #[allow(dead_code)]
    pub fn read_from<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let contents = std::fs::read(path).map_err(|e| e.to_string())?;
        bitcode::decode(&contents).map_err(|e| e.to_string())
    }

    /// Writes the ShellFlatVec to a file based on the format.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match Format::from(&path) {
            Format::Npy => self.write_npy(path),
            Format::Fasta => self.write_fasta(path),
        }
    }

    /// Writes a NPY file from the ShellFlatVec.
    pub fn write_npy<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            ShellFlatVec::String(_) => Err("Cannot write String data to NPY format".to_string()),
            ShellFlatVec::F32(data) => data.write_npy(&path),
            ShellFlatVec::F64(data) => data.write_npy(&path),
            ShellFlatVec::I8(data) => data.write_npy(&path),
            ShellFlatVec::I16(data) => data.write_npy(&path),
            ShellFlatVec::I32(data) => data.write_npy(&path),
            ShellFlatVec::I64(data) => data.write_npy(&path),
            ShellFlatVec::U8(data) => data.write_npy(&path),
            ShellFlatVec::U16(data) => data.write_npy(&path),
            ShellFlatVec::U32(data) => data.write_npy(&path),
            ShellFlatVec::U64(data) => data.write_npy(&path),
        }
    }

    /// Writes a FASTA file from the ShellFlatVec.
    pub fn write_fasta<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            ShellFlatVec::String(data) => fasta::write(path, data),
            _ => Err("Only String data can be written to FASTA format".to_string()),
        }
    }
}
