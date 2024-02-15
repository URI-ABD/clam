use distances::vectors;
use pyo3::prelude::*;

/// The `abd-distances` module implemented in Rust.
#[pymodule]
fn abd_distances(py: Python, m: &PyModule) -> PyResult<()> {
    register_vectors(py, m)?;
    Ok(())
}

fn register_vectors(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let vectors_module = PyModule::new(py, "vectors")?;
    vectors_module.add_function(wrap_pyfunction!(euclidean_f32, vectors_module)?)?;
    parent_module.add_submodule(vectors_module)?;
    Ok(())
}

/// Compute the Euclidean distance between two lists.
#[pyfunction]
fn euclidean_f32(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    Ok(vectors::euclidean(&a, &b))
}
