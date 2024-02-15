//! Python wrappers for distance functions implemented in Rust.

mod vectors;

use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(py: Python, m: &PyModule) -> PyResult<()> {
    vectors::register_vectors(py, m)?;
    Ok(())
}
