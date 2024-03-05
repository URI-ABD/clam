#![deny(clippy::correctness)]
#![warn(
    missing_docs,
    clippy::all,
    clippy::suspicious,
    clippy::style,
    clippy::complexity,
    clippy::perf,
    clippy::pedantic,
    clippy::nursery,
    clippy::missing_docs_in_private_items,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_lossless
)]
#![doc = include_str!("../README.md")]

use pyo3::prelude::*;

/// Array chunking
pub mod chunked_array;

/// The version of the crate.
pub const VERSION: &str = "0.1.0";

/// Formats the sum of two numbers as string.
#[must_use]
pub fn sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}

/// Formats the sum of two numbers as string.
///
/// # Errors
///
/// - If the sum of `a` and `b` is not representable as a string.
#[pyfunction]
#[pyo3(name = "sum_as_string")]
#[allow(clippy::unnecessary_wraps)]
pub fn py_sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

use pyo3::exceptions::PyBaseException;

/// Defines a chunked array for arbitrary type. This is necessary because pyo3 does not support
/// generics.
macro_rules! def_chunked_array {
    ($struct_name:ident, $t:ty, $module:expr) => {
        #[pyclass]
        ///.
        pub struct $struct_name {
            ///.
            pub ca: chunked_array::ChunkedArray<$t>,
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            ///.
            pub fn new(path: &str) -> PyResult<Self> {
                let ca = chunked_array::ChunkedArray::new(path)
                    .map_err(|e| PyBaseException::new_err(e))?;
                Ok(Self { ca })
            }

            ///.
            pub fn shape(&self) -> Vec<usize> {
                self.ca.shape.clone()
            }
        }

        // Add the class to the module
        $module.add_class::<$struct_name>()?;
    };
}

/// A Python module implemented in Rust.
///
/// # Errors
///
/// - If the module cannot be created.
/// - If some function cannot be added to the module.
#[pymodule]
#[allow(clippy::unnecessary_wraps)]
pub fn ndarray_chunked(_py: Python, m: &PyModule) -> PyResult<()> {
    def_chunked_array!(ChunkedArrayF32, f32, m);
    def_chunked_array!(ChunkedArrayF64, f64, m);
    def_chunked_array!(ChunkedArrayI32, i32, m);
    def_chunked_array!(ChunkedArrayI64, i64, m);
    def_chunked_array!(ChunkedArrayU32, u32, m);
    def_chunked_array!(ChunkedArrayU64, u64, m);
    Ok(())
}
