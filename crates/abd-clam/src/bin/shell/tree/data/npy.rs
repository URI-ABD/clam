//! Read and write NPY datasets.

/// Reads a vector dataset from a `npy` file at the specified path, returning a vector of vectors of the specified type.
pub fn read_npy_generic<P: AsRef<std::path::Path>, T: ndarray_npy::ReadableElement + Clone>(
    path: &P,
) -> Result<Vec<Vec<T>>, Box<dyn std::error::Error + Send + Sync>> {
    ndarray_npy::read_npy::<_, ndarray::Array2<T>>(path)
        .map(|arr| arr.outer_iter().map(|row| row.to_vec()).collect())
        .map_err(Into::into)
}

/// Writes a vector dataset to a `npy` file at the specified path.
#[expect(unused)]
pub fn write_npy_generic<P: AsRef<std::path::Path>, T: ndarray_npy::WritableElement + Clone>(
    path: &P,
    data: &[Vec<T>],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let shape = (data.len(), data[0].len());
    let flat_data = data.iter().flat_map(Clone::clone).collect();
    let arr = ndarray::Array2::from_shape_vec(shape, flat_data)?;
    ndarray_npy::write_npy(path, &arr)?;
    Ok(())
}
