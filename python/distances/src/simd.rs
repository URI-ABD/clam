//! SIMD accelerated distance functions for vectors.

use distances::simd;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

pub fn register(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let simd_module = PyModule::new(py, "simd")?;
    simd_module.add_function(wrap_pyfunction!(euclidean_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_sq_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_sq_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cosine_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cosine_f64, simd_module)?)?;
    parent_module.add_submodule(simd_module)?;
    Ok(())
}

/// Euclidean distance for 32-bit floating point vectors.
#[pyfunction]
fn euclidean_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(simd::euclidean_f32(a.as_slice()?, b.as_slice()?))
}

/// Euclidean distance for 64-bit floating point vectors.
#[pyfunction]
fn euclidean_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(simd::euclidean_f64(a.as_slice()?, b.as_slice()?))
}

/// Squared Euclidean distance for 32-bit floating point vectors.
#[pyfunction]
fn euclidean_sq_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(simd::euclidean_sq_f32(a.as_slice()?, b.as_slice()?))
}

/// Squared Euclidean distance for 64-bit floating point vectors.
#[pyfunction]
fn euclidean_sq_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(simd::euclidean_sq_f64(a.as_slice()?, b.as_slice()?))
}

/// Cosine distance for 32-bit floating point vectors.
#[pyfunction]
fn cosine_f32(a: PyReadonlyArray1<'_, f32>, b: PyReadonlyArray1<'_, f32>) -> PyResult<f32> {
    Ok(simd::cosine_f32(a.as_slice()?, b.as_slice()?))
}

/// Cosine distance for 64-bit floating point vectors.
#[pyfunction]
fn cosine_f64(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(simd::cosine_f64(a.as_slice()?, b.as_slice()?))
}
