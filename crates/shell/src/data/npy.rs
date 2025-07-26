//! Data types we might read from npy files.

use std::path::Path;

use abd_clam::FlatVec;

use super::ShellFlatVec;

pub enum NpyType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl NpyType {
    /// Reads an array from a NPY file.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellFlatVec, String> {
        for ty in Self::variants() {
            if let Ok(data) = ty.read_with_type(&path) {
                return Ok(data);
            }
        }
        Err(format!("Failed to read NPY file at path: {}", path.as_ref().display()))
    }

    fn variants() -> &'static [NpyType] {
        &[
            NpyType::F32,
            NpyType::F64,
            NpyType::I8,
            NpyType::I16,
            NpyType::I32,
            NpyType::I64,
            NpyType::U8,
            NpyType::U16,
            NpyType::U32,
            NpyType::U64,
        ]
    }

    /// Reads an array from a NPY file with a specific type.
    fn read_with_type<P: AsRef<Path>>(&self, path: P) -> Result<ShellFlatVec, String> {
        match self {
            Self::F32 => FlatVec::<Vec<f32>, usize>::read_npy(&path).map(ShellFlatVec::F32),
            Self::F64 => FlatVec::<Vec<f64>, usize>::read_npy(&path).map(ShellFlatVec::F64),
            Self::I8 => FlatVec::<Vec<i8>, usize>::read_npy(&path).map(ShellFlatVec::I8),
            Self::I16 => FlatVec::<Vec<i16>, usize>::read_npy(&path).map(ShellFlatVec::I16),
            Self::I32 => FlatVec::<Vec<i32>, usize>::read_npy(&path).map(ShellFlatVec::I32),
            Self::I64 => FlatVec::<Vec<i64>, usize>::read_npy(&path).map(ShellFlatVec::I64),
            Self::U8 => FlatVec::<Vec<u8>, usize>::read_npy(&path).map(ShellFlatVec::U8),
            Self::U16 => FlatVec::<Vec<u16>, usize>::read_npy(&path).map(ShellFlatVec::U16),
            Self::U32 => FlatVec::<Vec<u32>, usize>::read_npy(&path).map(ShellFlatVec::U32),
            Self::U64 => FlatVec::<Vec<u64>, usize>::read_npy(&path).map(ShellFlatVec::U64),
        }
    }
}
