//! Distance functions over sets of items.

use distances::strings::{hamming, levenshtein, nw_distance};
use pyo3::prelude::*;

/// Register the `strings` module with the Python interpreter.
pub fn register(pm: &Bound<'_, PyModule>) -> PyResult<()> {
    let strings_module = PyModule::new(pm.py(), "strings")?;
    strings_module.add_function(wrap_pyfunction!(hamming_distance, &strings_module)?)?;
    strings_module.add_function(wrap_pyfunction!(levenshtein_distance, &strings_module)?)?;
    strings_module.add_function(wrap_pyfunction!(needleman_wunsch_distance, &strings_module)?)?;
    pm.add_submodule(&strings_module)
}

/// Hamming distance for strings.
#[pyfunction]
#[pyo3(name = "hamming")]
fn hamming_distance(a: &str, b: &str) -> u64 {
    hamming(a, b)
}

/// Levenshtein distance for strings.
#[pyfunction]
#[pyo3(name = "levenshtein")]
fn levenshtein_distance(a: &str, b: &str) -> u64 {
    levenshtein(a, b)
}

/// Needleman-Wunsch distance for strings.
#[pyfunction]
#[pyo3(name = "needleman_wunsch")]
fn needleman_wunsch_distance(a: &str, b: &str) -> u64 {
    nw_distance(a, b)
}
