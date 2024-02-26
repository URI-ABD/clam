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

/// The version of the crate.
pub const VERSION: &str = "0.1.0";

///
pub mod chunked_array;

// use pyo3::prelude::*;

// /// Formats the sum of two numbers as string.
// #[must_use]
// pub fn sum_as_string(a: usize, b: usize) -> String {
//     (a + b).to_string()
// }

// /// Formats the sum of two numbers as string.
// ///
// /// # Errors
// ///
// /// - If the sum of `a` and `b` is not representable as a string.
// #[pyfunction]
// #[pyo3(name = "sum_as_string")]
// #[allow(clippy::unnecessary_wraps)]
// pub fn py_sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

// /// A Python module implemented in Rust.
// ///
// /// # Errors
// ///
// /// - If the module cannot be created.
// /// - If some function cannot be added to the module.
// #[pymodule]
// #[allow(clippy::unnecessary_wraps)]
// pub fn ndarray_chunked(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(py_sum_as_string, m)?)?;
//     Ok(())
// }
