//! Distance functions for vectors.

use distances::{vectors, Number};
use ndarray::parallel::prelude::*;
use numpy::{ndarray::Axis, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

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
    vectors_module.add_function(wrap_pyfunction!(cdist_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cdist_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cdist_u32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cdist_u64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cdist_i32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(cdist_i64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_f32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_f64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_u32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_u64, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_i32, vectors_module)?)?;
    vectors_module.add_function(wrap_pyfunction!(pdist_i64, vectors_module)?)?;
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

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_f32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f32>,
    b: PyReadonlyArray2<'_, f32>,
    metric: &str,
) -> PyResult<Py<PyArray2<f32>>> {
    let func = match metric {
        "chebyshev" => vectors::chebyshev,
        "euclidean" => vectors::euclidean,
        "sqeuclidean" => vectors::euclidean_sq,
        "l3_distance" => vectors::l3_norm,
        "l4_distance" => vectors::l4_norm,
        "manhattan" | "cityblock" => vectors::manhattan,
        "canberra" => vectors::canberra,
        "cosine" => vectors::cosine,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
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
        "chebyshev" => vectors::chebyshev,
        "euclidean" => vectors::euclidean,
        "sqeuclidean" => vectors::euclidean_sq,
        "l3_distance" => vectors::l3_norm,
        "l4_distance" => vectors::l4_norm,
        "manhattan" | "cityblock" => vectors::manhattan,
        "canberra" => vectors::canberra,
        "cosine" => vectors::cosine,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
}

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_u32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, u32>,
    b: PyReadonlyArray2<'_, u32>,
    metric: &str,
) -> PyResult<Py<PyArray2<f32>>> {
    let func = match metric {
        "braycurtis" => vectors::bray_curtis,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
}

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_u64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, u64>,
    b: PyReadonlyArray2<'_, u64>,
    metric: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let func = match metric {
        "braycurtis" => vectors::bray_curtis,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
}

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_i32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, i32>,
    b: PyReadonlyArray2<'_, i32>,
    metric: &str,
) -> PyResult<Py<PyArray2<u32>>> {
    let func = match metric {
        "hamming" => vectors::hamming,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
}

/// Computes the distance each pair of the two collections of inputs.
#[pyfunction]
fn cdist_i64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, i64>,
    b: PyReadonlyArray2<'_, i64>,
    metric: &str,
) -> PyResult<Py<PyArray2<u64>>> {
    let func = match metric {
        "hamming" => vectors::hamming,
        _ => return Err(PyValueError::new_err("Invalid metric")),
    };
    Ok(PyArray2::from_vec2(py, &cdist_helper(a, b, func))?.to_owned())
}

pub fn cdist_helper<T: Number + numpy::Element, U: Number>(
    a: PyReadonlyArray2<'_, T>,
    b: PyReadonlyArray2<'_, T>,
    func: fn(&[T], &[T]) -> U,
) -> Vec<Vec<U>> {
    let a = a.as_array();
    let b = b.as_array();
    a.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            b.axis_iter(Axis(0))
                .into_par_iter()
                .map(|col| func(row.as_slice().unwrap(), col.as_slice().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_f32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f32>,
    metric: &str,
) -> PyResult<Py<PyArray1<f32>>> {
    let func = match metric {
        "chebyshev" => vectors::chebyshev,
        "euclidean" => vectors::euclidean,
        "sqeuclidean" => vectors::euclidean_sq,
        "l3_distance" => vectors::l3_norm,
        "l4_distance" => vectors::l4_norm,
        "manhattan" | "cityblock" => vectors::manhattan,
        "canberra" => vectors::canberra,
        "cosine" => vectors::cosine,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_f64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, f64>,
    metric: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let func = match metric {
        "chebyshev" => vectors::chebyshev,
        "euclidean" => vectors::euclidean,
        "sqeuclidean" => vectors::euclidean_sq,
        "l3_distance" => vectors::l3_norm,
        "l4_distance" => vectors::l4_norm,
        "manhattan" | "cityblock" => vectors::manhattan,
        "canberra" => vectors::canberra,
        "cosine" => vectors::cosine,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_u32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, u32>,
    metric: &str,
) -> PyResult<Py<PyArray1<f32>>> {
    let func = match metric {
        "braycurtis" => vectors::bray_curtis,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_u64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, u64>,
    metric: &str,
) -> PyResult<Py<PyArray1<f64>>> {
    let func = match metric {
        "braycurtis" => vectors::bray_curtis,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_i32(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, i32>,
    metric: &str,
) -> PyResult<Py<PyArray1<u32>>> {
    let func = match metric {
        "hamming" => vectors::hamming,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

/// Computes the pairwise distances between all vectors in the collection.
#[pyfunction]
fn pdist_i64(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, i64>,
    metric: &str,
) -> PyResult<Py<PyArray1<u64>>> {
    let func = match metric {
        "hamming" => vectors::hamming,
        _ => return Err(PyValueError::new_err(format!("Invalid metric: {metric}"))),
    };
    Ok(pdist_helper(py, a, func))
}

pub fn pdist_helper<T: Number + numpy::Element, U: Number + numpy::Element>(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, T>,
    func: fn(&[T], &[T]) -> U,
) -> Py<PyArray1<U>> {
    let a = a.as_array();
    let result = a
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .flat_map(|(i, row)| {
            a.axis_iter(Axis(0))
                .into_par_iter()
                .skip(i + 1)
                .map(move |col| func(row.as_slice().unwrap(), col.as_slice().unwrap()))
        })
        .collect::<Vec<_>>();
    PyArray1::from_vec(py, result).to_owned()
}
