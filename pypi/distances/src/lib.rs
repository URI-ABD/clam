//! Python wrappers for distance functions implemented in Rust.

// TODO Najib: Address these in a future PR.
#![allow(
    clippy::unnested_or_patterns,
    clippy::implicit_clone,
    clippy::unwrap_used,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::missing_docs_in_private_items,
    clippy::used_underscore_items
)]

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
