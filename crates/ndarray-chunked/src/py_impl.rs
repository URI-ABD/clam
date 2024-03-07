use crate::chunked_array;
use numpy::convert::ToPyArray;
use pyo3::prelude::*;
use pyo3::types::PySlice;

/// Defines a chunked array for arbitrary type. This is necessary because pyo3 does not support
/// generics.
macro_rules! def_chunked_array {
    ($struct_name:ident, $fn_name:ident, $t:ty, $module:expr) => {
        #[pyclass]
        ///.
        struct $struct_name {
            ///.
            ca: chunked_array::ChunkedArray<$t>,
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            ///.
            fn new(path: &str) -> PyResult<Self> {
                let ca = chunked_array::ChunkedArray::new(path)
                    .map_err(|e| PyBaseException::new_err(e))?;
                Ok(Self { ca })
            }

            ///.
            fn shape(&self) -> Vec<usize> {
                self.ca.shape.clone()
            }

            ///.
            fn __getitem__(&self, py: Python, key: &PyTuple) -> pyo3::Py<PyArray<$t, Dim<IxDynImpl>>>{
                // We need to build the slice array from the tuple, converting the tuple elements to SliceInfoElems
                let mut slices = key.iter().enumerate().map(|(i, k)| {
                    // If it's an integer, just treat it like a normal index. No problem.
                    if PyAny::is_exact_instance_of::<PyInt>(k) {
                        let k = k.extract::<isize>().unwrap();
                        SliceInfoElem::Index(k)

                    // Otherwise we hav a slice, in which we want to build the slice from
                    // the pyslice
                    } else if PyAny::is_exact_instance_of::<PySlice>(k) {
                        let k = k.extract::<&PySlice>().unwrap();
                        let length = self.ca.shape[i];
                        let indices = k.indices(length as i64).unwrap();
                        SliceInfoElem::Slice {
                            start: indices.start,
                            end: Some(indices.stop),
                            step: indices.step,
                        }

                    // Otherwise (for now) we just panic
                    } else {
                        panic!("Invalid index type")
                    }
                }).collect::<Vec<SliceInfoElem>>();


                // It might be the case that the user only gives us k < n of the full
                // dimensionality for a slice. In this case, we need to fill the rest
                // with full slices.
                let n = self.ca.shape.len();
                if slices.len() < n {
                    slices.extend((slices.len()..n).map(|_| SliceInfoElem::Slice {
                        start: 0,
                        end: None,
                        step: 1,
                    }));
                }

                // Slice the actual array
                let sliced = self.ca.slice(&slices);
                sliced.to_pyarray(py).to_owned()
            }
        }

        // Add the class to the module
        $module.add_class::<$struct_name>()?;

        #[pyfunction]
        fn $fn_name(arr: &PyAny, chunk_along: usize, size: usize, path: &str) -> PyResult<()> {
            let arr = arr.extract::<&PyArray<$t, Dim<IxDynImpl>>>()
                .map_err(|e| PyBaseException::new_err(e))?;

            let arr = unsafe {
                arr.as_array()
            };

            chunked_array::ChunkedArray::chunk(&arr.view(), chunk_along, size, path)
                .map_err(|e| PyBaseException::new_err(e))?;
            Ok(())
        }
        $module.add_function(pyo3::wrap_pyfunction!($fn_name, $module)?)?;
    };
}

/// A Python module implemented in Rust.
///
/// # Errors
///
/// - If the module cannot be created.
/// - If some function cannot be added to the module.
#[pymodule]
#[allow(clippy::unnecessary_wraps, clippy::redundant_pub_crate)]
pub fn ndarray_chunked(_py: Python, m: &PyModule) -> PyResult<()> {
    use ndarray::IxDynImpl;
    use ndarray::{Dim, SliceInfoElem};
    use numpy::PyArray;
    use pyo3::exceptions::PyBaseException;
    use pyo3::types::{PyInt, PyTuple};

    def_chunked_array!(ChunkedArrayF32, chunk_f32, f32, m);
    // def_chunked_array!(ChunkedArrayF64, chunk_f64, f64, m);
    // def_chunked_array!(ChunkedArrayI32, chunk_i32, i32, m);
    // def_chunked_array!(ChunkedArrayI64, chunk_i64, i64, m);
    // def_chunked_array!(ChunkedArrayU32, chunk_u32, u32, m);
    // def_chunked_array!(ChunkedArrayU64, chunk_u64, u64, m);
    Ok(())
}
