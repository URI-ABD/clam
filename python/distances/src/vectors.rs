//! Distance functions for vectors.

use distances::vectors;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub fn register(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let vectors_module = PyModule::new(py, "vectors")?;
    vectors_module.add_function(wrap_pyfunction!(chebyshev_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(chebyshev_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(euclidean_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(euclidean_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(euclidean_sq_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(euclidean_sq_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(l3_norm_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(l3_norm_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(l4_norm_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(l4_norm_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(manhattan_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(manhattan_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(bray_curtis_u32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(bray_curtis_u64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(canberra_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(canberra_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cosine_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cosine_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(hamming_i32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(hamming_i64, vectors_module)?)?;
    parent_module.add_submodule(vectors_module)?;
    Ok(())
}

/// Chebyshev distance for 32-bit floating point vectors.
#[pyfunction]
fn chebyshev_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::chebyshev(a.as_slice()?, b.as_slice()?))
}

/// Chebyshev distance for 64-bit floating point vectors.
#[pyfunction]
fn chebyshev_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::chebyshev(a.as_slice()?, b.as_slice()?))
}

/// Euclidean distance for 32-bit floating point vectors.
#[pyfunction]
fn euclidean_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::euclidean(a.as_slice()?, b.as_slice()?))
}

/// Euclidean distance for 64-bit floating point vectors.
#[pyfunction]
fn euclidean_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::euclidean(a.as_slice()?, b.as_slice()?))
}

/// Squared Euclidean distance for 32-bit floating point vectors.
#[pyfunction]
fn euclidean_sq_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::euclidean_sq(a.as_slice()?, b.as_slice()?))
}

/// Squared Euclidean distance for 64-bit floating point vectors.
#[pyfunction]
fn euclidean_sq_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::euclidean_sq(a.as_slice()?, b.as_slice()?))
}

/// L3 norm for 32-bit floating point vectors.
#[pyfunction]
fn l3_norm_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::l3_norm(a.as_slice()?, b.as_slice()?))
}

/// L3 norm for 64-bit floating point vectors.
#[pyfunction]
fn l3_norm_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::l3_norm(a.as_slice()?, b.as_slice()?))
}

/// L4 norm for 32-bit floating point vectors.
#[pyfunction]
fn l4_norm_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::l4_norm(a.as_slice()?, b.as_slice()?))
}

/// L4 norm for 64-bit floating point vectors.
#[pyfunction]
fn l4_norm_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::l4_norm(a.as_slice()?, b.as_slice()?))
}

/// Manhattan distance for 32-bit floating point vectors.
#[pyfunction]
fn manhattan_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::manhattan(a.as_slice()?, b.as_slice()?))
}

/// Manhattan distance for 64-bit floating point vectors.
#[pyfunction]
fn manhattan_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::manhattan(a.as_slice()?, b.as_slice()?))
}

/// Bray-Curtis distance for 32-bit integer vectors.
#[pyfunction]
fn bray_curtis_u32(a: PyReadonlyArray1<'_, u32>, b: PyReadonlyArray1<'_, u32>) -> PyResult<f32> {
    Ok(vectors::bray_curtis(a.as_slice()?, b.as_slice()?))
}

/// Bray-Curtis distance for 64-bit integer vectors.
#[pyfunction]
fn bray_curtis_u64(a: PyReadonlyArray1<'_, u64>, b: PyReadonlyArray1<'_, u64>) -> PyResult<f64> {
    Ok(vectors::bray_curtis(a.as_slice()?, b.as_slice()?))
}

/// Canberra distance for 32-bit floating point vectors.
#[pyfunction]
fn canberra_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::canberra(a.as_slice()?, b.as_slice()?))
}

/// Canberra distance for 64-bit floating point vectors.
#[pyfunction]
fn canberra_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::canberra(a.as_slice()?, b.as_slice()?))
}

/// Cosine distance for 32-bit floating point vectors.
#[pyfunction]
fn cosine_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(vectors::cosine(a.as_slice()?, b.as_slice()?))
}

/// Cosine distance for 64-bit floating point vectors.
#[pyfunction]
fn cosine_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(vectors::cosine(a.as_slice()?, b.as_slice()?))
}

/// Hamming distance for 32-bit integer vectors.
#[pyfunction]
fn hamming_i32(a: PyReadonlyArray1<'_, i32>, b: PyReadonlyArray1<'_, i32>) -> PyResult<u32> {
    Ok(vectors::hamming(a.as_slice()?, b.as_slice()?))
}

/// Hamming distance for 64-bit integer vectors.
#[pyfunction]
fn hamming_i64(a: PyReadonlyArray1<'_, i64>, b: PyReadonlyArray1<'_, i64>) -> PyResult<u64> {
    Ok(vectors::hamming(a.as_slice()?, b.as_slice()?))
}
