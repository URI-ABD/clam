//! SIMD accelerated distance functions for vectors.

use distances::{simd, Number};
use numpy::{PyArray1, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::utils::{Scalar, Vector1, Vector2, _cdist, _pdist};

pub fn register(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let simd_module = PyModule::new(py, "simd")?;
    simd_module.add_function(wrap_pyfunction!(euclidean, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(sqeuclidean, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cosine, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(cdist, simd_module)?)?;
    simd_module.add_function(wrap_pyfunction!(pdist, simd_module)?)?;
    parent_module.add_submodule(simd_module)?;
    Ok(())
}

macro_rules! build_fn {
    ($name:tt, $name_f32:tt, $name_f64:tt) => {
        #[pyfunction]
        fn $name(a: Vector1, b: Vector1) -> PyResult<Scalar> {
            match (&a, &b) {
                // The types are the same
                (Vector1::F32(a), Vector1::F32(b)) => {
                    Ok(Scalar::F32(simd::$name_f32(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::F64(a), Vector1::F64(b)) => {
                    Ok(Scalar::F64(simd::$name_f64(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::U8(_), Vector1::U8(_)) => {
                    let a = a.cast::<f32>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    let b = b.cast::<f32>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    Ok(Scalar::F32(simd::$name_f32(a, b)))
                }
                (Vector1::U16(_), Vector1::U16(_)) => {
                    let a = a.cast::<f32>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    let b = b.cast::<f32>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    Ok(Scalar::F32(simd::$name_f32(a, b)))
                }
                (Vector1::U32(_), Vector1::U32(_)) => {
                    let a = a.cast::<f32>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    let b = b.cast::<f32>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    Ok(Scalar::F32(simd::$name_f32(a, b)))
                }
                (Vector1::U64(_), Vector1::U64(_)) => {
                    let a = a.cast::<f64>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    let b = b.cast::<f64>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Invalid type")),
                    }?;
                    Ok(Scalar::F64(simd::$name_f64(a, b)))
                }
                // The types are different
                (Vector1::F64(a), _) => {
                    let a = a.as_array();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    let b = b.cast::<f64>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F64(simd::$name_f64(a, b)))
                }
                (_, Vector1::F64(b)) => {
                    let a = a.cast::<f64>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    let b = b.as_array();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F64(simd::$name_f64(a, b)))
                }
                (Vector1::U64(_), _)
                | (Vector1::I64(_), _)
                | (_, Vector1::U64(_))
                | (_, Vector1::I64(_)) => {
                    let a = a.cast::<f64>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    let b = b.cast::<f64>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F64(simd::$name_f64(a, b)))
                }
                _ => {
                    let a = a.cast::<f32>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    let b = b.cast::<f32>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F32(simd::$name_f32(a, b)))
                }
            }
        }
    };
}

build_fn!(euclidean, euclidean_f32, euclidean_f64);
build_fn!(sqeuclidean, euclidean_sq_f32, euclidean_sq_f64);
build_fn!(cosine, cosine_f32, cosine_f64);

#[pyfunction]
fn cdist(py: Python<'_>, a: Vector2, b: Vector2, metric: &str) -> PyResult<Py<PyArray2<f64>>> {
    match (&a, &b) {
        // The types are the same
        (Vector2::F32(a), Vector2::F32(b)) => {
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?.to_owned())
        }
        (Vector2::F64(a), Vector2::F64(b)) => {
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?.to_owned())
        }
        (Vector2::U8(_), Vector2::U8(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::U16(_), Vector2::U16(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::U32(_), Vector2::U32(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::U64(_), Vector2::U64(_)) => {
            let a = a.cast::<f64>();
            let b = b.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::I8(_), Vector2::I8(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::I16(_), Vector2::I16(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::I32(_), Vector2::I32(_)) => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::I64(_), Vector2::I64(_)) => {
            let a = a.cast::<f64>();
            let b = b.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        // The types are different
        (Vector2::F64(a), _) => {
            let a = a.as_array();
            let b = b.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (_, Vector2::F64(b)) => {
            let a = a.cast::<f64>();
            let b = b.as_array();
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        (Vector2::U64(_), _)
        | (Vector2::I64(_), _)
        | (_, Vector2::U64(_))
        | (_, Vector2::I64(_)) => {
            let a = a.cast::<f64>();
            let b = b.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
        _ => {
            let a = a.cast::<f32>();
            let b = b.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
        }
    }
}

#[pyfunction]
fn pdist(py: Python<'_>, a: Vector2, metric: &str) -> PyResult<Py<PyArray1<f64>>> {
    match &a {
        Vector2::F32(a) => {
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.as_array(), metric))
        }
        Vector2::F64(a) => {
            let metric = _parse_metric_f64(metric)?;
            Ok(_pdist(py, a.as_array(), metric))
        }
        Vector2::U8(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::U16(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::U32(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::U64(_) => {
            let a = a.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::I8(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::I16(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::I32(_) => {
            let a = a.cast::<f32>();
            let metric = _parse_metric_f32(metric)?;
            let metric = |a: &[f32], b: &[f32]| metric(a, b).as_f64();
            Ok(_pdist(py, a.view(), metric))
        }
        Vector2::I64(_) => {
            let a = a.cast::<f64>();
            let metric = _parse_metric_f64(metric)?;
            Ok(_pdist(py, a.view(), metric))
        }
    }
}

#[allow(clippy::type_complexity)]
fn _parse_metric_f32(name: &str) -> PyResult<fn(&[f32], &[f32]) -> f32> {
    match name {
        "euclidean" => Ok(simd::euclidean_f32),
        "sqeuclidean" => Ok(simd::euclidean_sq_f32),
        "cosine" => Ok(simd::cosine_f32),
        _ => Err(PyValueError::new_err(format!("Unknown metric: {name}"))),
    }
}

#[allow(clippy::type_complexity)]
fn _parse_metric_f64(name: &str) -> PyResult<fn(&[f64], &[f64]) -> f64> {
    match name {
        "euclidean" => Ok(simd::euclidean_f64),
        "sqeuclidean" => Ok(simd::euclidean_sq_f64),
        "cosine" => Ok(simd::cosine_f64),
        _ => Err(PyValueError::new_err(format!("Unknown metric: {name}"))),
    }
}
