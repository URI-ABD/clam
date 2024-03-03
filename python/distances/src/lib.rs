//! Python wrappers for distance functions implemented in Rust.

pub(crate) mod simd;
pub(crate) mod strings;
pub(crate) mod typeless_vectors;
mod utils;

use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(py: Python, m: &PyModule) -> PyResult<()> {
    simd::register(py, m)?;
    strings::register(py, m)?;
    typeless_vectors::register(py, m)?;
    Ok(())
}
