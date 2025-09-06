//! Data types we might read from npy files.

use std::path::Path;

use ndarray::Array2;

use super::ShellData;

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
    pub fn read<P: AsRef<Path>>(path: P) -> Result<ShellData, String> {
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
    fn read_with_type<P: AsRef<Path>>(&self, path: P) -> Result<ShellData, String> {
        match self {
            Self::F32 => read_npy(path).map(ShellData::F32),
            Self::F64 => read_npy(path).map(ShellData::F64),
            Self::I8 => read_npy(path).map(ShellData::I8),
            Self::I16 => read_npy(path).map(ShellData::I16),
            Self::I32 => read_npy(path).map(ShellData::I32),
            Self::I64 => read_npy(path).map(ShellData::I64),
            Self::U8 => read_npy(path).map(ShellData::U8),
            Self::U16 => read_npy(path).map(ShellData::U16),
            Self::U32 => read_npy(path).map(ShellData::U32),
            Self::U64 => read_npy(path).map(ShellData::U64),
        }
    }
}

pub fn read_npy<P: AsRef<Path>, T: ndarray_npy::ReadableElement + Clone>(path: P) -> Result<Vec<Vec<T>>, String> {
    let arr = ndarray_npy::read_npy::<_, Array2<T>>(path).map_err(|e| e.to_string())?;
    let vecs = arr.outer_iter().map(|row| row.to_vec()).collect();
    Ok(vecs)
}

pub fn write_npy<P: AsRef<Path>, T: ndarray_npy::WritableElement + Clone>(path: P, data: &[Vec<T>]) -> Result<(), String> {
    let (n_rows, n_cols) = (data.len(), data.first().map_or(0, |row| row.len()));
    let flat_data = data.iter().flatten().cloned().collect::<Vec<_>>();
    let arr = Array2::from_shape_vec((n_rows, n_cols), flat_data).map_err(|e| e.to_string())?;
    ndarray_npy::write_npy(path, &arr).map_err(|e| e.to_string())
}

pub fn read_npy_n<P: AsRef<Path>, T: ndarray_npy::ReadableElement + Clone + Copy, const N: usize>(path: P) -> Result<Vec<[T; N]>, String> {
    let arr = ndarray_npy::read_npy::<_, Array2<T>>(path).map_err(|e| e.to_string())?;
    from_array2(&arr)
}

pub fn write_npy_n<P: AsRef<Path>, T: ndarray_npy::WritableElement + Clone, const N: usize>(path: P, data: &[[T; N]]) -> Result<(), String> {
    let arr = to_array2(data)?;
    ndarray_npy::write_npy(path, &arr).map_err(|e| e.to_string())
}

pub fn to_array2<T: Clone, const N: usize>(data: &[[T; N]]) -> Result<Array2<T>, String> {
    Array2::from_shape_vec((data.len(), N), data.iter().flat_map(|row| row.to_vec()).collect()).map_err(|e| e.to_string())
}

pub fn from_array2<T: Clone + Copy, const N: usize>(arr: &Array2<T>) -> Result<Vec<[T; N]>, String> {
    if arr.ncols() != N {
        return Err(format!("Array has {} columns, expected {}", arr.ncols(), N));
    }
    let mut data = Vec::with_capacity(arr.nrows());
    for row in arr.outer_iter() {
        let mut fixed_row = [row[0]; N];
        for (i, &item) in row.iter().enumerate().take(N) {
            fixed_row[i] = item;
        }
        data.push(fixed_row);
    }
    Ok(data)
}
