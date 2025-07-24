//! Reading `hdf5` files from the `ann-benchmarks` suite.

use abd_clam::{Dataset, FlatVec};
use distances::Number;

macro_rules! read_ty {
    ($inp_dir:expr, $name:expr, $($ty:ty),*) => {
        $(
            let data = read_helper::<_, $ty>($inp_dir, $name);
            if data.is_ok() {
                return data;
            }
            let err = data.err();
            ftlog::info!("{err:?}");
            println!("{err:?}");
        )*
    };
}

pub fn read<P: AsRef<std::path::Path>>(inp_dir: &P, name: &str) -> Result<FlatVec<Vec<f32>, usize>, String> {
    read_ty!(inp_dir, name, f32, f64, i32, i64, u32, u64);
    Err(format!("Unsupported data type for {name}"))
}

/// Reads a dataset in `hdf5` format from the `ann-benchmarks` suite.
fn read_helper<P: AsRef<std::path::Path>, T: Number + hdf5::H5Type + Clone>(
    inp_dir: &P,
    name: &str,
) -> Result<FlatVec<Vec<f32>, usize>, String> {
    let npy_path = inp_dir.as_ref().join(format!("{name}.npy"));
    if npy_path.exists() {
        ftlog::info!("Reading npy data from {npy_path:?}...");
        return FlatVec::<Vec<f32>, usize>::read_npy(&npy_path);
    }

    let hdf5_path = inp_dir.as_ref().join(format!("{name}.hdf5"));
    if !hdf5_path.exists() {
        return Err(format!("{hdf5_path:?} does not exist"));
    }

    ftlog::info!("Reading hdf5 data from {hdf5_path:?}...");

    // TODO: Deal with the flattened flag.
    let flattened = false;

    let data = bench_utils::ann_benchmarks::read::<_, T>(&hdf5_path, flattened)?;
    let items = data.train;
    let (min_len, max_len) = items.iter().fold((usize::MAX, 0), |(min, max), item| {
        let len = item.len();
        (Ord::min(min, len), Ord::max(max, len))
    });
    let data = FlatVec::new(items)?
        .with_name(name)
        .with_dim_lower_bound(min_len)
        .with_dim_upper_bound(max_len);

    // Convert to `f32` for the dimensionality reduction.
    let data = data.transform_items(|v| v.iter().map(|x| x.as_f32()).collect::<Vec<_>>());
    if min_len == max_len {
        ftlog::info!("Writing hdf5 data in npy format to {npy_path:?}...");
        data.write_npy(&npy_path)?;
    }

    Ok(data)
}
