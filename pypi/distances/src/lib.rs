//! Python wrappers for distance functions implemented in Rust.

#![expect(clippy::match_same_arms)]

pub(crate) mod simd;
pub(crate) mod strings;
mod utils;
pub(crate) mod vectors;

use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(m: &Bound<'_, PyModule>) -> PyResult<()> {
    simd::register(m)?;
    strings::register(m)?;
    vectors::register(m)?;
    Ok(())
}
