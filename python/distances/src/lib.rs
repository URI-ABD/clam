//! Python wrappers for distance functions implemented in Rust.

pub(crate) mod simd;
pub(crate) mod strings;
mod utils;
pub(crate) mod vectors;

use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(py: Python, m: &PyModule) -> PyResult<()> {
    simd::register(py, m)?;
    strings::register(py, m)?;
    vectors::register(py, m)?;
    Ok(())
}
