//! Helpers for the Python wrapper.

use distances::{vectors, Number};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2};
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
    U8(PyReadonlyArray1<'py, u8>),
    U16(PyReadonlyArray1<'py, u16>),
    U32(PyReadonlyArray1<'py, u32>),
    U64(PyReadonlyArray1<'py, u64>),
    I8(PyReadonlyArray1<'py, i8>),
    I16(PyReadonlyArray1<'py, i16>),
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
}

impl<'a> Vector1<'a> {
    pub fn cast<T: Number + numpy::Element>(&self) -> Array1<T> {
        match self {
            Vector1::F32(a) => a.as_array().mapv(T::from),
            Vector1::F64(a) => a.as_array().mapv(T::from),
            Vector1::U8(a) => a.as_array().mapv(T::from),
            Vector1::U16(a) => a.as_array().mapv(T::from),
            Vector1::U32(a) => a.as_array().mapv(T::from),
            Vector1::U64(a) => a.as_array().mapv(T::from),
            Vector1::I8(a) => a.as_array().mapv(T::from),
            Vector1::I16(a) => a.as_array().mapv(T::from),
            Vector1::I32(a) => a.as_array().mapv(T::from),
            Vector1::I64(a) => a.as_array().mapv(T::from),
        }
    }
}

impl<'a> FromPyObject<'a> for Vector1<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, f32>>() {
            Ok(Vector1::F32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, f64>>() {
            Ok(Vector1::F64(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, u8>>() {
            Ok(Vector1::U8(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, u16>>() {
            Ok(Vector1::U16(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, u32>>() {
            Ok(Vector1::U32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, u64>>() {
            Ok(Vector1::U64(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, i8>>() {
            Ok(Vector1::I8(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, i16>>() {
            Ok(Vector1::I16(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, i32>>() {
            Ok(Vector1::I32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray1<'_, i64>>() {
            Ok(Vector1::I64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
        }
    }
}

pub enum Vector2<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
    U8(PyReadonlyArray2<'py, u8>),
    U16(PyReadonlyArray2<'py, u16>),
    U32(PyReadonlyArray2<'py, u32>),
    U64(PyReadonlyArray2<'py, u64>),
    I8(PyReadonlyArray2<'py, i8>),
    I16(PyReadonlyArray2<'py, i16>),
    I32(PyReadonlyArray2<'py, i32>),
    I64(PyReadonlyArray2<'py, i64>),
}

impl<'a> Vector2<'a> {
    pub fn cast<T: Number + numpy::Element>(&self) -> Array2<T> {
        match self {
            Vector2::F32(a) => a.as_array().mapv(T::from),
            Vector2::F64(a) => a.as_array().mapv(T::from),
            Vector2::U8(a) => a.as_array().mapv(T::from),
            Vector2::U16(a) => a.as_array().mapv(T::from),
            Vector2::U32(a) => a.as_array().mapv(T::from),
            Vector2::U64(a) => a.as_array().mapv(T::from),
            Vector2::I8(a) => a.as_array().mapv(T::from),
            Vector2::I16(a) => a.as_array().mapv(T::from),
            Vector2::I32(a) => a.as_array().mapv(T::from),
            Vector2::I64(a) => a.as_array().mapv(T::from),
        }
    }
}

impl<'a> FromPyObject<'a> for Vector2<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, f32>>() {
            Ok(Vector2::F32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, f64>>() {
            Ok(Vector2::F64(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, u8>>() {
            Ok(Vector2::U8(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, u16>>() {
            Ok(Vector2::U16(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, u32>>() {
            Ok(Vector2::U32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, u64>>() {
            Ok(Vector2::U64(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, i8>>() {
            Ok(Vector2::I8(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, i16>>() {
            Ok(Vector2::I16(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, i32>>() {
            Ok(Vector2::I32(a))
        } else if let Ok(a) = ob.extract::<PyReadonlyArray2<'_, i64>>() {
            Ok(Vector2::I64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn parse_metric<T: Number>(metric: &str) -> PyResult<fn(&[T], &[T]) -> f64> {
    match metric.to_lowercase().as_str() {
        "braycurtis" => Ok(vectors::bray_curtis),
        "canberra" => Ok(vectors::canberra),
        "chebyshev" => Ok(_chebyshev),
        "euclidean" => Ok(vectors::euclidean),
        "sqeuclidean" => Ok(vectors::euclidean_sq),
        "manhattan" | "cityblock" => Ok(_manhattan),
        "cosine" => Ok(vectors::cosine),
        _ => Err(PyValueError::new_err(format!("Unknown metric: {}", metric))),
    }
}

pub fn _chebyshev<T: Number, U: Number>(a: &[T], b: &[T]) -> U {
    let d = vectors::chebyshev(a, b);
    U::from(d)
}

pub fn _manhattan<T: Number, U: Number>(a: &[T], b: &[T]) -> U {
    let d = vectors::manhattan(a, b);
    U::from(d)
}

pub fn _cdist<T, U, F>(
    a: ndarray::ArrayView2<T>,
    b: ndarray::ArrayView2<T>,
    metric: F,
) -> Vec<Vec<U>>
where
    T: Number + numpy::Element,
    U: Number + numpy::Element,
    F: Fn(&[T], &[T]) -> U + Send + Sync + Copy,
{
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

pub fn _pdist<T, U, F>(py: Python<'_>, a: ndarray::ArrayView2<T>, metric: F) -> Py<PyArray1<U>>
where
    T: Number + numpy::Element,
    U: Number + numpy::Element,
    F: Fn(&[T], &[T]) -> U + Send + Sync + Copy,
{
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
