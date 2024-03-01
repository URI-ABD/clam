//! Distance functions for vectors.

use distances::{simd, vectors, Number};
use ndarray::parallel::prelude::*;
use numpy::{ndarray::Axis, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

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

enum Vector1<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

impl<'a> FromPyObject<'a> for Vector1<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, f32>>() {
            Ok(Vector1::F32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, f64>>() {
            Ok(Vector1::F64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
        }
    }
}

/// Compute the Chebyshev distance between two vectors.
#[pyfunction]
fn chebyshev(a: Vector1, b: Vector1) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(vectors::chebyshev(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => Ok(vectors::chebyshev(a.as_slice()?, b.as_slice()?)),
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

/// Compute the Euclidean distance between two vectors.
#[pyfunction]
fn euclidean(a: Vector1, b: Vector1) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(simd::euclidean_f32(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => Ok(simd::euclidean_f64(a.as_slice()?, b.as_slice()?)),
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

/// Compute the squared Euclidean distance between two vectors.
#[pyfunction]
fn sqeuclidean(a: Vector1, b: Vector1) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(simd::euclidean_sq_f32(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => {
            Ok(simd::euclidean_sq_f64(a.as_slice()?, b.as_slice()?))
        }
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

/// Compute the Manhattan distance between two vectors.
#[pyfunction]
fn manhattan(a: Vector1, b: Vector1) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(vectors::manhattan(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => Ok(vectors::manhattan(a.as_slice()?, b.as_slice()?)),
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

#[pyfunction]
fn minkowski(a: Vector1, b: Vector1, p: i32) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(vectors::minkowski::<_, f32>(p)(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => {
            Ok(vectors::minkowski(p)(a.as_slice()?, b.as_slice()?))
        }
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

/// Compute the cosine distance between two vectors.
#[pyfunction]
fn cosine(a: Vector1, b: Vector1) -> PyResult<f64> {
    match (a, b) {
        (Vector1::F32(a), Vector1::F32(b)) => {
            Ok(simd::cosine_f32(a.as_slice()?, b.as_slice()?).as_f64())
        }
        (Vector1::F64(a), Vector1::F64(b)) => Ok(simd::cosine_f64(a.as_slice()?, b.as_slice()?)),
        _ => Err(PyTypeError::new_err("Mismatched types")),
    }
}

enum Vector2<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl<'a> FromPyObject<'a> for Vector2<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, f32>>() {
            Ok(Vector2::F32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, f64>>() {
            Ok(Vector2::F64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
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
                let metric = parse_func(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a, b, metric))?.to_owned())
            }
            (Vector2::F64(a), Vector2::F64(b)) => {
                let metric = parse_func(metric)?;
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
                let metric = parse_func(metric)?;
                Ok(_pdist(py, a, metric))
            }
            Vector2::F64(a) => {
                let metric = parse_func(metric)?;
                Ok(_pdist(py, a, metric))
            }
        },
    }
}

#[allow(clippy::type_complexity)]
fn parse_func<T: Number>(metric: &str) -> PyResult<fn(&[T], &[T]) -> f64> {
    match metric.to_lowercase().as_str() {
        "chebyshev" => Ok(_chebyshev),
        "euclidean" => Ok(vectors::euclidean),
        "sqeuclidean" => Ok(vectors::euclidean_sq),
        "manhattan" | "cityblock" => Ok(_manhattan),
        "cosine" => Ok(vectors::cosine),
        _ => Err(PyValueError::new_err(format!("Unknown metric: {}", metric))),
    }
}

fn _chebyshev<T: Number>(a: &[T], b: &[T]) -> f64 {
    vectors::chebyshev(a, b).as_f64()
}

fn _manhattan<T: Number>(a: &[T], b: &[T]) -> f64 {
    vectors::manhattan(a, b).as_f64()
}

fn _cdist<T: Number + numpy::Element, F: Fn(&[T], &[T]) -> f64 + Send + Sync + Copy>(
    a: PyReadonlyArray2<'_, T>,
    b: PyReadonlyArray2<'_, T>,
    metric: F,
) -> Vec<Vec<f64>> {
    let a = a.as_array();
    let b = b.as_array();
    a.axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            b.axis_iter(Axis(0))
                .into_par_iter()
                .map(|col| metric(row.as_slice().unwrap(), col.as_slice().unwrap()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn _pdist<T: Number + numpy::Element, F: Fn(&[T], &[T]) -> f64 + Send + Sync + Copy>(
    py: Python<'_>,
    a: PyReadonlyArray2<'_, T>,
    metric: F,
) -> Py<PyArray1<f64>> {
    let a = a.as_array();
    let result = a
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .flat_map(|(i, row)| {
            a.axis_iter(Axis(0))
                .into_par_iter()
                .skip(i + 1)
                .map(move |col| metric(row.as_slice().unwrap(), col.as_slice().unwrap()))
        })
        .collect::<Vec<_>>();
    PyArray1::from_vec(py, result).to_owned()
}
