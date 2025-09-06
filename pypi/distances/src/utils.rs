//! Helpers for the Python wrapper.

use std::convert::Infallible;

use distances::{Number, vectors};
use ndarray::{Array1, Array2, parallel::prelude::*};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ndarray::Axis};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    types::PyFloat,
};

/// The types of scalar data we support.
pub enum Scalar {
    /// The data is a single f32 value.
    F32(f32),
    /// The data is a single f64 value.
    F64(f64),
}

impl<'a> FromPyObject<'a> for Scalar {
    #[expect(clippy::option_if_let_else)] // It's just cleaner this way.
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<f32>() {
            Ok(Self::F32(a))
        } else if let Ok(a) = ob.extract::<f64>() {
            Ok(Self::F64(a))
        } else {
            Err(PyTypeError::new_err("Invalid type"))
        }
    }
}

impl<'py> IntoPyObject<'py> for Scalar {
    type Target = PyFloat;

    type Output = Bound<'py, Self::Target>;

    type Error = Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Self::F32(a) => a.into_pyobject(py),
            Self::F64(a) => a.into_pyobject(py),
        }
    }
}

/// The types of 1D data we support.
pub enum Vector1<'py> {
    /// The data is a 1D array of f32.
    F32(PyReadonlyArray1<'py, f32>),
    /// The data is a 1D array of f64.
    F64(PyReadonlyArray1<'py, f64>),
    /// The data is a 1D array of u8.
    U8(PyReadonlyArray1<'py, u8>),
    /// The data is a 1D array of u16.
    U16(PyReadonlyArray1<'py, u16>),
    /// The data is a 1D array of u32.
    U32(PyReadonlyArray1<'py, u32>),
    /// The data is a 1D array of u64.
    U64(PyReadonlyArray1<'py, u64>),
    /// The data is a 1D array of i8.
    I8(PyReadonlyArray1<'py, i8>),
    /// The data is a 1D array of i16.
    I16(PyReadonlyArray1<'py, i16>),
    /// The data is a 1D array of i32.
    I32(PyReadonlyArray1<'py, i32>),
    /// The data is a 1D array of i64.
    I64(PyReadonlyArray1<'py, i64>),
}

impl Vector1<'_> {
    /// Casts the elements of the array to a specified numeric type.
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
    #[expect(clippy::option_if_let_else)] // It's just cleaner this way.
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
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

/// The types of 2D data we support.
pub enum Vector2<'py> {
    /// The data is a 2D array of f32.
    F32(PyReadonlyArray2<'py, f32>),
    /// The data is a 2D array of f64.
    F64(PyReadonlyArray2<'py, f64>),
    /// The data is a 2D array of u8.
    U8(PyReadonlyArray2<'py, u8>),
    /// The data is a 2D array of u16.
    U16(PyReadonlyArray2<'py, u16>),
    /// The data is a 2D array of u32.
    U32(PyReadonlyArray2<'py, u32>),
    /// The data is a 2D array of u64.
    U64(PyReadonlyArray2<'py, u64>),
    /// The data is a 2D array of i8.
    I8(PyReadonlyArray2<'py, i8>),
    /// The data is a 2D array of i16.
    I16(PyReadonlyArray2<'py, i16>),
    /// The data is a 2D array of i32.
    I32(PyReadonlyArray2<'py, i32>),
    /// The data is a 2D array of i64.
    I64(PyReadonlyArray2<'py, i64>),
}

impl Vector2<'_> {
    /// Casts the elements of the array to a specified numeric type.
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
    #[expect(clippy::option_if_let_else)] // It's just cleaner this way.
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
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

/// Converts a metric name to a distance function.
#[allow(clippy::type_complexity)]
pub fn parse_metric<T: Number>(metric: &str) -> PyResult<fn(&[T], &[T]) -> f64> {
    match metric.to_lowercase().as_str() {
        "braycurtis" => Ok(vectors::bray_curtis),
        "canberra" => Ok(vectors::canberra),
        "chebyshev" => Ok(chebyshev_generic),
        "euclidean" => Ok(vectors::euclidean),
        "sqeuclidean" => Ok(vectors::euclidean_sq),
        "manhattan" | "cityblock" => Ok(manhattan_generic),
        "cosine" => Ok(vectors::cosine),
        _ => Err(PyValueError::new_err(format!("Unknown metric: {metric}"))),
    }
}

/// A wrapper for the Chebyshev distance function.
pub fn chebyshev_generic<T: Number, U: Number>(a: &[T], b: &[T]) -> U {
    let d = vectors::chebyshev(a, b);
    U::from(d)
}

/// A wrapper for the Manhattan distance function.
pub fn manhattan_generic<T: Number, U: Number>(a: &[T], b: &[T]) -> U {
    let d = vectors::manhattan(a, b);
    U::from(d)
}

/// Computes the pairwise distances between rows of two 2D arrays using a
/// generic metric function, returning a 2D array of distances.
#[expect(clippy::expect_used)]
pub fn cdist_generic<'py, T, U, F>(py: Python<'py>, a: ndarray::ArrayView2<T>, b: ndarray::ArrayView2<T>, metric: F) -> Bound<'py, PyArray2<U>>
where
    T: Number + numpy::Element,
    U: Number + numpy::Element,
    F: Fn(&[T], &[T]) -> U + Send + Sync + Copy,
{
    let result = a
        .axis_iter(Axis(0))
        .into_par_iter()
        .flat_map(|row| {
            b.axis_iter(Axis(0)).into_par_iter().map(move |col| {
                metric(
                    row.as_slice().expect("ndarrays should be contiguous"),
                    col.as_slice().expect("ndarrays should be contiguous"),
                )
            })
        })
        .collect::<Vec<_>>();
    let shape = (a.nrows(), b.nrows());
    let arr = ndarray::Array2::from_shape_vec(shape, result).expect("Failed to create Array2 from shape and data");
    PyArray2::from_array(py, &arr)
}

/// Computes the pairwise distances between rows of a 2D array using a generic
/// metric function, returning a flattened 1D array of distances from the lower
/// triangular part of the distance matrix.
#[expect(clippy::expect_used)]
pub fn pdist_generic<'py, T, U, F>(py: Python<'py>, a: ndarray::ArrayView2<T>, metric: F) -> Bound<'py, PyArray1<U>>
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
            a.axis_iter(Axis(0)).into_par_iter().skip(i + 1).map(move |col| {
                metric(
                    row.as_slice().expect("ndarrays should be contiguous"),
                    col.as_slice().expect("ndarrays should be contiguous"),
                )
            })
        })
        .collect::<Vec<_>>();
    PyArray1::from_vec(py, result)
}
