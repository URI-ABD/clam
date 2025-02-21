//! Reading `npy` files.

use abd_clam::FlatVec;
use distances::Number;

/// Reads a numpy file and returns the data as a `FlatVec` of `f32`.
pub fn read<P: AsRef<std::path::Path>>(path: &P) -> Result<FlatVec<Vec<f32>, usize>, String> {
    ftlog::info!("Reading npy data from {:?}", path.as_ref());

    let data = FlatVec::<Vec<f32>, usize>::read_npy(path);
    if data.is_ok() {
        return data;
    }

    let data = FlatVec::<Vec<f64>, usize>::read_npy(path);
    if data.is_ok() {
        return data.map(|data| data.transform_items(|v| v.iter().map(|x| x.as_f32()).collect::<Vec<_>>()));
    }

    Err(format!("Failed to read {:?}", path.as_ref()))
}
