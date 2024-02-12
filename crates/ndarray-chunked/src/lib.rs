use pyo3::prelude::*;

pub const VERSION: &str = "0.1.0";

pub fn sum_as_string(a: usize, b: usize) -> String {
    (a + b).to_string()
}

/// Formats the sum of two numbers as string.
#[pyfunction]
#[pyo3(name = "sum_as_string")]
fn py_sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn ndarray_chunked(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_sum_as_string, m)?)?;
    Ok(())
}
