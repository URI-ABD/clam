//! Distance functions for vectors.

use distances::{vectors, Number};
use numpy::{PyArray1, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*};

use crate::utils::Scalar;

use super::utils::{parse_metric, Vector1, Vector2, _cdist, _chebyshev, _manhattan, _pdist};

pub fn register(py: Python<'_>, pm: &PyModule) -> PyResult<()> {
    let m = PyModule::new(py, "vectors")?;
    m.add_function(wrap_pyfunction!(braycurtis, m)?)?;
    m.add_function(wrap_pyfunction!(canberra, m)?)?;
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

macro_rules! build_fn {
    ($name:tt, $func:expr) => {
        #[pyfunction]
        fn $name(a: Vector1, b: Vector1) -> PyResult<Scalar> {
            match (&a, &b) {
                // The types are the same
                (Vector1::F32(a), Vector1::F32(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::F64(a), Vector1::F64(b)) => {
                    Ok(Scalar::F64($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::U8(a), Vector1::U8(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::U16(a), Vector1::U16(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::U32(a), Vector1::U32(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::U64(a), Vector1::U64(b)) => {
                    Ok(Scalar::F64($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::I8(a), Vector1::I8(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::I16(a), Vector1::I16(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::I32(a), Vector1::I32(b)) => {
                    Ok(Scalar::F32($func(a.as_slice()?, b.as_slice()?)))
                }
                (Vector1::I64(a), Vector1::I64(b)) => {
                    Ok(Scalar::F64($func(a.as_slice()?, b.as_slice()?)))
                }
                // The types are different
                (Vector1::F64(a), _) => {
                    let b = b.cast::<f64>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F64($func(a.as_slice()?, b)))
                }
                (_, Vector1::F64(b)) => {
                    let a = a.cast::<f64>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F64($func(a, b.as_slice()?)))
                }
                (Vector1::U64(_), _) | (Vector1::I64(_), _) => {
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
                    Ok(Scalar::F64($func(a, b)))
                }
                (_, Vector1::U64(_)) | (_, Vector1::I64(_)) => {
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
                    Ok(Scalar::F64($func(a, b)))
                }
                (Vector1::F32(a), _) => {
                    let b = b.cast::<f32>();
                    let b = match b.as_slice() {
                        Some(b) => Ok(b),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F32($func(a.as_slice()?, b)))
                }
                (_, Vector1::F32(b)) => {
                    let a = a.cast::<f32>();
                    let a = match a.as_slice() {
                        Some(a) => Ok(a),
                        None => Err(PyValueError::new_err("Non-contiguous array")),
                    }?;
                    Ok(Scalar::F32($func(a, b.as_slice()?)))
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
                    Ok(Scalar::F32($func(a, b)))
                }
            }
        }
    };
}

build_fn!(braycurtis, vectors::bray_curtis);
build_fn!(canberra, vectors::canberra);
build_fn!(chebyshev, _chebyshev);
build_fn!(euclidean, vectors::euclidean);
build_fn!(sqeuclidean, vectors::euclidean_sq);
build_fn!(manhattan, _manhattan);
build_fn!(cosine, vectors::cosine);

#[pyfunction]
fn minkowski(a: Vector1, b: Vector1, p: i32) -> PyResult<Scalar> {
    match (&a, &b) {
        // The types are the same
        (Vector1::F32(a), Vector1::F32(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::F64(a), Vector1::F64(b)) => Ok(Scalar::F64(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::U8(a), Vector1::U8(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::U16(a), Vector1::U16(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::U32(a), Vector1::U32(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::U64(a), Vector1::U64(b)) => Ok(Scalar::F64(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::I8(a), Vector1::I8(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::I16(a), Vector1::I16(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::I32(a), Vector1::I32(b)) => Ok(Scalar::F32(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        (Vector1::I64(a), Vector1::I64(b)) => Ok(Scalar::F64(vectors::minkowski(p)(
            a.as_slice()?,
            b.as_slice()?,
        ))),
        // The types are different
        (Vector1::F64(a), _) => {
            let b = b.cast::<f64>();
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F64(vectors::minkowski(p)(a.as_slice()?, b)))
        }
        (_, Vector1::F64(b)) => {
            let a = a.cast::<f64>();
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F64(vectors::minkowski(p)(a, b.as_slice()?)))
        }
        (Vector1::U64(_), _) | (Vector1::I64(_), _) => {
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
            Ok(Scalar::F64(vectors::minkowski(p)(a, b)))
        }
        (_, Vector1::U64(_)) | (_, Vector1::I64(_)) => {
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
            Ok(Scalar::F64(vectors::minkowski(p)(a, b)))
        }
        (Vector1::F32(a), _) => {
            let b = b.cast::<f32>();
            let b = match b.as_slice() {
                Some(b) => Ok(b),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::minkowski(p)(a.as_slice()?, b)))
        }
        (_, Vector1::F32(b)) => {
            let a = a.cast::<f32>();
            let a = match a.as_slice() {
                Some(a) => Ok(a),
                None => Err(PyValueError::new_err("Non-contiguous array")),
            }?;
            Ok(Scalar::F32(vectors::minkowski(p)(a, b.as_slice()?)))
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
            Ok(Scalar::F32(vectors::minkowski(p)(a, b)))
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
            match (&a, &b) {
                // The types are the same
                (Vector2::F32(a), Vector2::F32(b)) => {
                    let metric =
                        |a: &[f32], b: &[f32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::F64(a), Vector2::F64(b)) => {
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::U8(a), Vector2::U8(b)) => {
                    let metric =
                        |a: &[u8], b: &[u8]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::U16(a), Vector2::U16(b)) => {
                    let metric =
                        |a: &[u16], b: &[u16]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::U32(a), Vector2::U32(b)) => {
                    let metric =
                        |a: &[u32], b: &[u32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::U64(a), Vector2::U64(b)) => {
                    let metric = |a: &[u64], b: &[u64]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::I8(a), Vector2::I8(b)) => {
                    let metric =
                        |a: &[i8], b: &[i8]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::I16(a), Vector2::I16(b)) => {
                    let metric =
                        |a: &[i16], b: &[i16]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::I32(a), Vector2::I32(b)) => {
                    let metric =
                        |a: &[i32], b: &[i32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::I64(a), Vector2::I64(b)) => {
                    let metric = |a: &[i64], b: &[i64]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                // The types are different
                (Vector2::F64(a), _) => {
                    let b = b.cast::<f64>();
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.view(), metric))?
                            .to_owned(),
                    )
                }
                (_, Vector2::F64(b)) => {
                    let a = a.cast::<f64>();
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.view(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                (Vector2::U64(_), _) | (Vector2::I64(_), _) => {
                    let a = a.cast::<f64>();
                    let b = b.cast::<f64>();
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
                }
                (_, Vector2::U64(_)) | (_, Vector2::I64(_)) => {
                    let a = a.cast::<f64>();
                    let b = b.cast::<f64>();
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
                }
                (Vector2::F32(a), _) => {
                    let b = b.cast::<f32>();
                    let metric = |a: &[f32], b: &[f32]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.as_array(), b.view(), metric))?
                            .to_owned(),
                    )
                }
                (_, Vector2::F32(b)) => {
                    let a = a.cast::<f32>();
                    let metric = |a: &[f32], b: &[f32]| vectors::minkowski(p)(a, b);
                    Ok(
                        PyArray2::from_vec2(py, &_cdist(a.view(), b.as_array(), metric))?
                            .to_owned(),
                    )
                }
                _ => {
                    let a = a.cast::<f32>();
                    let b = b.cast::<f32>();
                    let metric = |a: &[f32], b: &[f32]| vectors::minkowski(p)(a, b);
                    Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
                }
            }
        }
        _ => match (&a, &b) {
            // The types are the same
            (Vector2::F32(a), Vector2::F32(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::F64(a), Vector2::F64(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::U8(a), Vector2::U8(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::U16(a), Vector2::U16(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::U32(a), Vector2::U32(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::U64(a), Vector2::U64(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::I8(a), Vector2::I8(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::I16(a), Vector2::I16(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::I32(a), Vector2::I32(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            (Vector2::I64(a), Vector2::I64(b)) => {
                let metric = parse_metric(metric)?;
                Ok(
                    PyArray2::from_vec2(py, &_cdist(a.as_array(), b.as_array(), metric))?
                        .to_owned(),
                )
            }
            // The types are different
            (Vector2::F64(a), _) => {
                let b = b.cast::<f64>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.as_array(), b.view(), metric))?.to_owned())
            }
            (_, Vector2::F64(b)) => {
                let a = a.cast::<f64>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.as_array(), metric))?.to_owned())
            }
            (Vector2::U64(_), _) | (Vector2::I64(_), _) => {
                let a = a.cast::<f64>();
                let b = b.cast::<f64>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
            }
            (_, Vector2::U64(_)) | (_, Vector2::I64(_)) => {
                let a = a.cast::<f64>();
                let b = b.cast::<f64>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
            }
            (Vector2::F32(a), _) => {
                let b = b.cast::<f32>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.as_array(), b.view(), metric))?.to_owned())
            }
            (_, Vector2::F32(b)) => {
                let a = a.cast::<f32>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.as_array(), metric))?.to_owned())
            }
            _ => {
                let a = a.cast::<f32>();
                let b = b.cast::<f32>();
                let metric = parse_metric(metric)?;
                Ok(PyArray2::from_vec2(py, &_cdist(a.view(), b.view(), metric))?.to_owned())
            }
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
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::F64(a) => {
                    let metric = |a: &[f64], b: &[f64]| vectors::minkowski(p)(a, b);
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::U8(a) => {
                    let metric =
                        |a: &[u8], b: &[u8]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::U16(a) => {
                    let metric =
                        |a: &[u16], b: &[u16]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::U32(a) => {
                    let metric =
                        |a: &[u32], b: &[u32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::U64(a) => {
                    let metric = |a: &[u64], b: &[u64]| vectors::minkowski(p)(a, b);
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::I8(a) => {
                    let metric =
                        |a: &[i8], b: &[i8]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::I16(a) => {
                    let metric =
                        |a: &[i16], b: &[i16]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::I32(a) => {
                    let metric =
                        |a: &[i32], b: &[i32]| vectors::minkowski::<_, f32>(p)(a, b).as_f64();
                    Ok(_pdist(py, a.as_array(), metric))
                }
                Vector2::I64(a) => {
                    let metric = |a: &[i64], b: &[i64]| vectors::minkowski(p)(a, b);
                    Ok(_pdist(py, a.as_array(), metric))
                }
            }
        }
        _ => match a {
            Vector2::F32(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::F64(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::U8(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::U16(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::U32(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::U64(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::I8(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::I16(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::I32(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
            Vector2::I64(a) => {
                let metric = parse_metric(metric)?;
                Ok(_pdist(py, a.as_array(), metric))
            }
        },
    }
}
