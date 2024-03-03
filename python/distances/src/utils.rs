//! Helpers for the Python wrapper.

use distances::{vectors, Number};
use ndarray::parallel::prelude::*;
use numpy::{ndarray::Axis, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
};

pub enum Scalar {
    F32(f32),
    F64(f64),
}

impl<'a> FromPyObject<'a> for Scalar {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<f32>() {
            Ok(Scalar::F32(a))
        } else if let Ok(a) = ob.extract::<f64>() {
            Ok(Scalar::F64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
        }
    }
}

impl IntoPy<PyObject> for Scalar {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Scalar::F32(a) => a.into_py(py),
            Scalar::F64(a) => a.into_py(py),
        }
    }
}

pub enum Vector1<'py> {
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

pub enum Vector2<'py> {
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

#[allow(clippy::type_complexity)]
pub fn parse_metric<T: Number>(metric: &str) -> PyResult<fn(&[T], &[T]) -> f64> {
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

pub fn _cdist<T, U, F>(
    a: PyReadonlyArray2<'_, T>,
    b: PyReadonlyArray2<'_, T>,
    metric: F,
) -> Vec<Vec<U>>
where
    T: Number + numpy::Element,
    U: Number + numpy::Element,
    F: Fn(&[T], &[T]) -> U + Send + Sync + Copy,
{
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

pub fn _pdist<T, U, F>(py: Python<'_>, a: PyReadonlyArray2<'_, T>, metric: F) -> Py<PyArray1<U>>
where
    T: Number + numpy::Element,
    U: Number + numpy::Element,
    F: Fn(&[T], &[T]) -> U + Send + Sync + Copy,
{
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
