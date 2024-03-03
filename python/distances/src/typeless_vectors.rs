//! Distance functions for vectors.

use distances::{vectors, Number};
use numpy::{PyArray1, PyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

use crate::utils::Scalar;

use super::utils::{parse_metric, Vector1, Vector2, _cdist, _pdist};

pub fn register(py: Python<'_>, pm: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "typeless_vectors")?;
    m.add_function(wrap_pyfunction!(chebyshev, m)?)?;
    m.add_function(wrap_pyfunction!(euclidean, m)?)?;
    m.add_function(wrap_pyfunction!(sqeuclidean, m)?)?;
    m.add_function(wrap_pyfunction!(manhattan, m)?)?;
    m.add_function(wrap_pyfunction!(minkowski, m)?)?;
    m.add_function(wrap_pyfunction!(cosine, m)?)?;
    m.add_function(wrap_pyfunction!(cdist, m)?)?;
    m.add_function(wrap_pyfunction!(pdist, m)?)?;
    pm.add_submodule(m)?;
    Ok(())
}

/// Compute the Chebyshev distance between two vectors.
#[pyfunction]
fn chebyshev(a: Vector1, b: Vector1) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::chebyshev(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::chebyshev(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::chebyshev(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::chebyshev(a, b)))
        }
    }
}

/// Compute the Euclidean distance between two vectors.
#[pyfunction]
fn euclidean(a: Vector1, b: Vector1) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::euclidean::<_, f32>(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::euclidean(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::euclidean(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::euclidean(a, b)))
        }
    }
}

/// Compute the squared Euclidean distance between two vectors.
#[pyfunction]
fn sqeuclidean(a: Vector1, b: Vector1) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::euclidean_sq::<_, f32>(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::euclidean_sq(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::euclidean_sq(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::euclidean_sq(a, b)))
        }
    }
}

/// Compute the Manhattan distance between two vectors.
#[pyfunction]
fn manhattan(a: Vector1, b: Vector1) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::manhattan(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::manhattan(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::manhattan(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::manhattan(a, b)))
        }
    }
}

#[pyfunction]
fn minkowski(a: Vector1, b: Vector1, p: i32) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::minkowski(p)(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::minkowski(p)(a, b)))
        }
    }
}

/// Compute the cosine distance between two vectors.
#[pyfunction]
fn cosine(a: Vector1, b: Vector1) -> PyResult<Scalar> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(Scalar::F32(vectors::cosine(a.as_slice()?, b.as_slice()?)))
        }
        (Vector1::F64(a), Vector1::F64(b)) => {
            Ok(Scalar::F64(vectors::cosine(a.as_slice()?, b.as_slice()?)))
        }
        (Vector1::F32(a), Vector1::F64(b)) => {
            let a = a.as_slice()?;
            let b = b.as_array().mapv(<f32 as Number>::from);
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::cosine(a, b)))
        }
        (Vector1::F64(a), Vector1::F32(b)) => {
            let a = a.as_array().mapv(<f32 as Number>::from);
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            let b = b.as_slice()?;
            Ok(Scalar::F32(vectors::cosine(a, b)))
        }
    }
}

/// Compute the pairwise distances between two collections of vectors.
#[pyfunction]
fn cdist(
    py: Python<'_>,
    a: Vector2,
    b: Vector2,
    metric: &str,
    p: Option<i32>,
) -> PyResult<Py<PyArray2<f64>>> {
    match p {
        Some(p) => {
            if metric.to_lowercase() != "minkowski" {
                return Err(PyValueError::new_err(
                    "p is only valid for Minkowski distance",
                ));
            }
            match (a, b) {
                (Vector2::F32(a), Vector2::F32(b)) => {
                    let metric =
                        |a: &[f32], b: &[f32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(PyArray2::from_vec2(py, &_cdist(a, b, metric))?.to_owned())
                }
                (Vector2::F64(a), Vector2::F64(b)) => {
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(PyArray2::from_vec2(py, &_cdist(a, b, metric))?.to_owned())
                }
                _ => Err(PyTypeError::new_err("Mismatched types")),
            }
        }
        _ => match (a, b) {
            (Vector2::F32(a), Vector2::F32(b)) => {
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a, b, metric))?.to_owned())
            }
            (Vector2::F64(a), Vector2::F64(b)) => {
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a, b, metric))?.to_owned())
            }
            _ => Err(PyTypeError::new_err("Mismatched types")),
        },
    }
}

/// Compute the pairwise distances between all vectors in a collection.
#[pyfunction]
fn pdist(py: Python<'_>, a: Vector2, metric: &str, p: Option<i32>) -> PyResult<Py<PyArray1<f64>>> {
    match p {
        Some(p) => {
            if metric.to_lowercase() != "minkowski" {
                return Err(PyValueError::new_err(
                    "p is only valid for Minkowski distance",
                ));
            }
            match a {
                Vector2::F32(a) => {
                    let metric =
                        |a: &[f32], b: &[f32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a, metric))
                }
                Vector2::F64(a) => {
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(_pdist(py, a, metric))
                }
            }
        }
        _ => match a {
            Vector2::F32(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a, metric))
            }
            Vector2::F64(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a, metric))
            }
        },
    }
}
