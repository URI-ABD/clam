//! Utilities for reading, writing, and generating data.

use std::path::Path;

mod gen_random;
mod neighborhood_aware;
mod vec_metric;
mod wasserstein;

use abd_clam::FlatVec;
pub use gen_random::gen_random;
pub use neighborhood_aware::NeighborhoodAware;
pub use vec_metric::VecMetric;

/// Read data from the given path or generate random data.
pub fn read_or_generate<P: AsRef<Path>>(
    path: Option<P>,
    num_inliers: Option<usize>,
    dimensionality: Option<usize>,
    inlier_mean: Option<f32>,
    inlier_std: Option<f32>,
    seed: Option<u64>,
) -> Result<FlatVec<Vec<f32>, usize>, String> {
    let data = if let Some(path) = path {
        let path = path.as_ref();
        if !path.exists() {
            return Err(format!("{path:?} does not exist"));
        }

        let ext = path.extension().and_then(|s| s.to_str());
        ext.map_or_else(
            || Err(format!("File extension not found in {path:?}")),
            |ext| match ext {
                "csv" => FlatVec::read_csv(&path),
                "npy" => FlatVec::read_npy(&path),
                _ => Err(format!("Unsupported file extension: {ext}. Must be `csv` or `npy`")),
            },
        )
    } else {
        // Check that all the required parameters are provided.
        let car = num_inliers.ok_or("num_inliers must be provided")?;
        let dim = dimensionality.ok_or("dimensionality must be provided")?;
        let mean = inlier_mean.ok_or("inlier_mean must be provided")?;
        let std = inlier_std.ok_or("inlier_std must be provided")?;

        let data = gen_random(mean, std, car, dim, seed);
        FlatVec::new_array(data)
    }?;

    let dim = data.items().first().map(Vec::len).ok_or("No instances found")?;
    if data.items().iter().any(|v| v.len() != dim) {
        return Err("Inconsistent dimensionality".to_string());
    }

    Ok(data.with_dim_lower_bound(dim).with_dim_upper_bound(dim))
}
