//! Reading `hdf5` files from the `ann-benchmarks` suite.

use abd_clam::{Dataset, FlatVec};
use distances::Number;
use rayon::prelude::*;

/// `ann-benchmarks` datasets are expected to have this many neighbors.
const ANN_NUM_NEIGHBORS: usize = 100;

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

    let data = AnnDataset::<T>::read(&hdf5_path, flattened)?;
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

/// A helper for storing training and query data for `ann-benchmarks`'s datasets
/// along with the ground truth nearest neighbors and distances.
#[expect(dead_code)]
pub struct AnnDataset<T> {
    /// The data to use for clustering.
    pub train: Vec<Vec<T>>,
    /// The queries to use for search.
    pub queries: Vec<Vec<T>>,
    /// The true neighbors of each query, given as a tuple of:
    /// * index into `train`, and
    /// * distance to the query.
    pub neighbors: Vec<Vec<(usize, f32)>>,
}

impl AnnDataset<f32> {
    /// Augment the dataset by adding noisy copies of the data.
    #[must_use]
    #[expect(dead_code)]
    pub fn augment(mut self, multiplier: usize, error_rate: f32) -> Self {
        ftlog::info!("Augmenting dataset to {multiplier}x...");
        self.train = symagen::augmentation::augment_data(&self.train, multiplier, error_rate);

        self
    }

    /// Generate a random dataset with the given metric.
    #[must_use]
    #[expect(dead_code)]
    pub fn gen_random(cardinality: usize, n_copies: usize, dimensionality: usize, n_queries: usize, seed: u64) -> Self {
        let train = (0..n_copies)
            .into_par_iter()
            .flat_map(|i| {
                let seed = seed + i.as_u64();
                symagen::random_data::random_tabular_seedable(cardinality, dimensionality, -1.0, 1.0, seed)
            })
            .collect::<Vec<_>>();
        let queries = symagen::random_data::random_tabular_seedable(
            n_queries,
            dimensionality,
            -1.0,
            1.0,
            seed + n_copies.as_u64(),
        );

        Self {
            train,
            queries,
            neighbors: Vec::new(),
        }
    }
}

impl<T: hdf5::H5Type + Clone> AnnDataset<T> {
    /// Reads an `ann-benchmarks` dataset from the given path.
    ///
    /// The dataset is expected be an HDF5 group with the following members:
    ///
    /// * `train`: The data to use for clustering.
    /// * `test`: The queries to use for search.
    /// * `neighbors`: The true neighbors of each query, given as indices into
    ///   `train`.
    /// * `distances`: The distances from each query to its true neighbors.
    ///
    /// If the dataset is flattened (as for `Kosarak` and `MovieLens10M`), then we
    /// expect the following additional members:
    ///
    /// * `size_train`: The lengths of each inner vector in `train`.
    /// * `size_test`: The lengths of each inner vector in `test`.
    ///
    /// # Arguments
    ///
    /// * `path`: The path to the dataset.
    /// * `flattened`: Whether to read the `train` and `test` datasets as flattened
    ///   vectors.
    ///
    /// # Returns
    ///
    /// The dataset, if it was read successfully.
    ///
    /// # Errors
    ///
    /// * If the dataset is not readable.
    /// * If the dataset is not in the expected format.
    pub fn read<P: AsRef<std::path::Path>>(path: &P, flattened: bool) -> Result<Self, String> {
        let path = path.as_ref();

        let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
        ftlog::info!("Opened file: {}", path.display());

        ftlog::info!("Reading raw train and test datasets...");
        let train_raw = file.dataset("train").map_err(|e| e.to_string())?;
        let test_raw = file.dataset("test").map_err(|e| e.to_string())?;

        let (train, queries) = if flattened {
            // Read flattened dataset
            let train = train_raw.read_raw::<T>().map_err(|e| e.to_string())?;
            let size_train = file
                .dataset("size_train")
                .map_err(|e| e.to_string())?
                .read_raw::<usize>()
                .map_err(|e| e.to_string())?;

            // Read flattened queries
            let queries = test_raw.read_raw::<T>().map_err(|e| e.to_string())?;
            let size_test = file
                .dataset("size_test")
                .map_err(|e| e.to_string())?
                .read_raw::<usize>()
                .map_err(|e| e.to_string())?;

            ftlog::info!("Un-flattening datasets...");
            (un_flatten(train, &size_train)?, un_flatten(queries, &size_test)?)
        } else {
            // Read 2d-array dataset
            let train = train_raw
                .read_2d::<T>()
                .map_err(|e| e.to_string())?
                .rows()
                .into_iter()
                .map(|r| r.to_vec())
                .collect();

            // Read 2d-array queries
            let queries = test_raw
                .read_2d::<T>()
                .map_err(|e| e.to_string())?
                .rows()
                .into_iter()
                .map(|r| r.to_vec())
                .collect();

            (train, queries)
        };
        ftlog::info!("Parsed {} train and {} query items.", train.len(), queries.len());

        ftlog::info!("Reading true neighbors and distances...");
        let neighbors = {
            let neighbors = file.dataset("neighbors").map_err(|e| e.to_string())?;
            let neighbors = neighbors.read_raw::<usize>().map_err(|e| e.to_string())?;

            let distances = file.dataset("distances").map_err(|e| e.to_string())?;
            let distances = distances.read_raw::<f32>().map_err(|e| e.to_string())?;

            if neighbors.len() != distances.len() {
                return Err(format!(
                    "`neighbors` and `distances` have different lengths! {} vs {}",
                    neighbors.len(),
                    distances.len()
                ));
            }

            let neighbors = neighbors.into_iter().zip(distances).collect();
            let sizes = vec![ANN_NUM_NEIGHBORS; queries.len()];
            un_flatten(neighbors, &sizes)?
        };

        Ok(Self {
            train,
            queries,
            neighbors,
        })
    }
}

/// Un-flattens a vector of data into a vector of vectors.
///
/// # Arguments
///
/// * `data` - The data to un-flatten.
/// * `sizes` - The sizes of the inner vectors.
///
/// # Returns
///
/// A vector of vectors where each inner vector has the size specified in `sizes`.
///
/// # Errors
///
/// * If the number of elements in `data` is not equal to the sum of the elements in `sizes`.
fn un_flatten<T>(data: Vec<T>, sizes: &[usize]) -> Result<Vec<Vec<T>>, String> {
    let num_elements: usize = sizes.iter().sum();
    if data.len() != num_elements {
        return Err(format!(
            "Incorrect number of elements. Expected: {num_elements}. Found: {}.",
            data.len()
        ));
    }

    let mut iter = data.into_iter();
    let mut items = Vec::with_capacity(sizes.len());
    for &s in sizes {
        let mut inner = Vec::with_capacity(s);
        for _ in 0..s {
            inner.push(iter.next().ok_or("Not enough elements!")?);
        }
        items.push(inner);
    }
    Ok(items)
}
