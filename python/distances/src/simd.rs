//! SIMD accelerated distance functions for vectors.

use distances::simd;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

pub fn register(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let simd_module = PyModule::new(py, "simd")?;
    simd_module.add_function(wrap_pyfunction!(euclidean_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_sq_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(euclidean_sq_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cosine_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cosine_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cdist_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cdist_f64, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(pdist_f32, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(pdist_f64, simd_module)?)?;
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

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_f32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
    metric: &str,
) -> PyResult<Py<PyArray2<f32>>> {
    let func = match metric {
        "euclidean" => simd::euclidean_f32,
        "sqeuclidean" => simd::euclidean_sq_f32,
        "cosine" => simd::cosine_f32,
        _ => return Err(PyValueError::new_err("Invalid metric")),
    };
    Ok(
        PyArray2::from_vec2(py, &super::utils::_cdist(a.as_array(), b.as_array(), func))?
            .to_owned(),
    )
}

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_f64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray2<'_, f64>,
    metric: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let func = match metric {
        "euclidean" => simd::euclidean_f64,
        "sqeuclidean" => simd::euclidean_sq_f64,
        "cosine" => simd::cosine_f64,
        _ => return Err(PyValueError::new_err("Invalid metric")),
    };
    Ok(
        PyArray2::from_vec2(py, &super::utils::_cdist(a.as_array(), b.as_array(), func))?
            .to_owned(),
    )
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_f32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f32>,
    metric: &str,
) -> PyResult<Py<PyArray1<f32>>> {
    let func = match metric {
        "euclidean" => simd::euclidean_f32,
        "sqeuclidean" => simd::euclidean_sq_f32,
        "cosine" => simd::cosine_f32,
        _ => return Err(PyValueError::new_err("Invalid metric")),
    };
    Ok(super::utils::_pdist(py, a.as_array(), func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_f64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f64>,
    metric: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let func = match metric {
        "euclidean" => simd::euclidean_f64,
        "sqeuclidean" => simd::euclidean_sq_f64,
        "cosine" => simd::cosine_f64,
        _ => return Err(PyValueError::new_err("Invalid metric")),
    };
    Ok(super::utils::_pdist(py, a.as_array(), func))
}
