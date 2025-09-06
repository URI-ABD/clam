//! Data formats supported in the CLI.

use std::path::Path;
use std::str::FromStr;

use abd_clam::musals::Sequence;
use rand::prelude::*;
use rayon::prelude::*;

pub mod fasta;
pub mod npy;

pub use fasta::MusalsSequence;

/// Supported output formats for data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum OutputFormat {
    /// Json format.
    Json,
    /// Yaml format.
    Yaml,
}

impl OutputFormat {
    /// Returns the file extension associated with the input format.
    pub const fn extension(&self) -> &str {
        match self {
            Self::Json => "json",
            Self::Yaml => "yaml",
        }
    }

    /// Creates a new OutputFormat from a Path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let ext = path
            .as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| format!("Failed to get file extension from path {:?}", path.as_ref()))?;

        match ext.to_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            _ => Err(format!("Unknown format: '{ext}'. Use 'json' or 'yaml'/'yml'.")),
        }
    }

    /// Reads the input data from the given path based on the input format.
    pub fn read<P, T>(path: P) -> Result<T, String>
    where
        P: AsRef<Path> + core::fmt::Debug,
        T: for<'de> serde::Deserialize<'de>,
    {
        match Self::from_path(&path)? {
            Self::Json => {
                let content = std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file {path:?}: {e}"))?;
                serde_json::from_str(&content).map_err(|e| format!("Failed to parse JSON from file {path:?}: {e}"))
            }
            Self::Yaml => {
                let content = std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file {path:?}: {e}"))?;
                serde_yaml::from_str(&content).map_err(|e| format!("Failed to parse YAML from file {path:?}: {e}"))
            }
        }
    }

    /// Writes the given data to the specified path in the output format.
    pub fn write<P, T>(path: P, data: &T) -> Result<(), String>
    where
        P: AsRef<Path> + core::fmt::Debug,
        T: serde::Serialize,
    {
        let content = match Self::from_path(&path)? {
            Self::Json => serde_json::to_string(data).map_err(|e| format!("Failed to serialize data to JSON: {e}"))?,
            Self::Yaml => serde_yaml::to_string(data).map_err(|e| format!("Failed to serialize data to YAML: {e}"))?,
        };

        std::fs::write(&path, content).map_err(|e| format!("Failed to write file {path:?}: {e}"))?;

        Ok(())
    }

    /// Writes the given data to the specified path in the output format using pretty formatting.
    pub fn write_pretty<P, T>(path: P, data: &T) -> Result<(), String>
    where
        P: AsRef<Path> + core::fmt::Debug,
        T: serde::Serialize,
    {
        let content = match Self::from_path(&path)? {
            Self::Json => serde_json::to_string_pretty(data).map_err(|e| format!("Failed to serialize data to JSON: {e}"))?,
            Self::Yaml => serde_yaml::to_string(data).map_err(|e| format!("Failed to serialize data to YAML: {e}"))?,
        };

        std::fs::write(&path, content).map_err(|e| format!("Failed to write file {path:?}: {e}"))?;

        Ok(())
    }
}

impl core::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
        }
    }
}

/// Supported formats for input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum InputFormat {
    /// Npy array format.
    Npy,
    /// FASTA format.
    Fasta,
}

impl InputFormat {
    /// Returns the file extension associated with the input format.
    pub const fn extension(&self) -> &str {
        match self {
            Self::Npy => "npy",
            Self::Fasta => "fasta",
        }
    }

    /// Creates a new InputFormat from a Path.
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let ext = path
            .as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| format!("Failed to get file extension from path {:?}", path.as_ref()))?;

        match ext.to_lowercase().as_str() {
            "npy" => Ok(Self::Npy),
            "fasta" => Ok(Self::Fasta),
            _ => Err(format!("Unknown format: '{ext}'. Use 'npy' or 'fasta'.")),
        }
    }

    /// Reads the input data from the given path based on the input format.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to the input data file.
    /// * `remove_gaps` - Whether to remove gaps from sequence data (only applicable for FASTA format).
    /// * `sample_size` - Optional size of the subsample of the input data to read.
    /// * `rng` - A mutable reference to a random number generator for shuffling the data.
    pub fn read<P: AsRef<Path> + core::fmt::Debug, R: rand::Rng>(
        path: P,
        remove_gaps: bool,
        sample_size: Option<usize>,
        rng: &mut R,
    ) -> Result<ShellData, String> {
        let mut data = match Self::from_path(&path)? {
            Self::Npy => npy::NpyType::read(path),
            Self::Fasta => fasta::read(path, remove_gaps).map(ShellData::String),
        }?;
        data.shuffle(rng);
        if let Some(size) = sample_size {
            data.truncate(size);
        }
        Ok(data)
    }

    /// Writes the given data to the specified path in the input format.
    pub fn write<P: AsRef<Path> + core::fmt::Debug>(path: P, data: &ShellData) -> Result<(), String> {
        data.write(path)
    }
}

impl core::fmt::Display for InputFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Npy => write!(f, "npy"),
            Self::Fasta => write!(f, "fasta"),
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

#[derive(databuf::Encode, databuf::Decode, serde::Deserialize, serde::Serialize)]
pub enum ShellData {
    /// Vec of sequences and their metadata from FASTA files.
    String(Vec<(String, MusalsSequence)>),
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

impl core::fmt::Display for ShellData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ShellData::String(_) => write!(f, "String data"),
            ShellData::F32(_) => write!(f, "F32 data"),
            ShellData::F64(_) => write!(f, "F64 data"),
            ShellData::I8(_) => write!(f, "I8 data"),
            ShellData::I16(_) => write!(f, "I16 data"),
            ShellData::I32(_) => write!(f, "I32 data"),
            ShellData::I64(_) => write!(f, "I64 data"),
            ShellData::U8(_) => write!(f, "U8 data"),
            ShellData::U16(_) => write!(f, "U16 data"),
            ShellData::U32(_) => write!(f, "U32 data"),
            ShellData::U64(_) => write!(f, "U64 data"),
        }
    }
}

impl ShellData {
    /// Remove gaps from sequence data, and errors if not string data.
    pub fn without_gaps(self, gaps: &[u8]) -> Result<Self, String> {
        match self {
            Self::String(mut data) => {
                data.par_iter_mut().for_each(|(_, seq)| {
                    *seq = seq.without_bytes(gaps);
                });
                Ok(Self::String(data))
            }
            _ => Err("Gap removal is only supported for string data.".to_string()),
        }
    }

    /// Create a slice of a ShellData from start to end indices.
    pub fn slice(&self, start: usize, end: usize) -> Self {
        match self {
            Self::String(data) => Self::String(data[start..end].to_vec()),
            Self::F32(data) => Self::F32(data[start..end].to_vec()),
            Self::F64(data) => Self::F64(data[start..end].to_vec()),
            Self::I8(data) => Self::I8(data[start..end].to_vec()),
            Self::I16(data) => Self::I16(data[start..end].to_vec()),
            Self::I32(data) => Self::I32(data[start..end].to_vec()),
            Self::I64(data) => Self::I64(data[start..end].to_vec()),
            Self::U8(data) => Self::U8(data[start..end].to_vec()),
            Self::U16(data) => Self::U16(data[start..end].to_vec()),
            Self::U32(data) => Self::U32(data[start..end].to_vec()),
            Self::U64(data) => Self::U64(data[start..end].to_vec()),
        }
    }

    /// Shuffles the ShellData in place using the provided RNG.
    pub fn shuffle<R: rand::Rng>(&mut self, rng: &mut R) {
        match self {
            Self::String(data) => data.shuffle(rng),
            Self::F32(data) => data.shuffle(rng),
            Self::F64(data) => data.shuffle(rng),
            Self::I8(data) => data.shuffle(rng),
            Self::I16(data) => data.shuffle(rng),
            Self::I32(data) => data.shuffle(rng),
            Self::I64(data) => data.shuffle(rng),
            Self::U8(data) => data.shuffle(rng),
            Self::U16(data) => data.shuffle(rng),
            Self::U32(data) => data.shuffle(rng),
            Self::U64(data) => data.shuffle(rng),
        }
    }

    /// Truncates the ShellData to the specified size.
    pub fn truncate(&mut self, size: usize) {
        match self {
            Self::String(data) => data.truncate(size),
            Self::F32(data) => data.truncate(size),
            Self::F64(data) => data.truncate(size),
            Self::I8(data) => data.truncate(size),
            Self::I16(data) => data.truncate(size),
            Self::I32(data) => data.truncate(size),
            Self::I64(data) => data.truncate(size),
            Self::U8(data) => data.truncate(size),
            Self::U16(data) => data.truncate(size),
            Self::U32(data) => data.truncate(size),
            Self::U64(data) => data.truncate(size),
        }
    }

    /// Writes the ShellData to a file at the given path.
    pub fn write<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        match self {
            Self::String(data) => fasta::write(path, data),
            Self::F32(data) => npy::write_npy(path, data),
            Self::F64(data) => npy::write_npy(path, data),
            Self::I8(data) => npy::write_npy(path, data),
            Self::I16(data) => npy::write_npy(path, data),
            Self::I32(data) => npy::write_npy(path, data),
            Self::I64(data) => npy::write_npy(path, data),
            Self::U8(data) => npy::write_npy(path, data),
            Self::U16(data) => npy::write_npy(path, data),
            Self::U32(data) => npy::write_npy(path, data),
            Self::U64(data) => npy::write_npy(path, data),
        }
    }
}
