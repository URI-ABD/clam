//! Reading from the hdf5 datasets provided by the `ann-benchmarks` repository on GitHub.

use std::path::Path;

use hdf5::H5Type;

/// The number of neighbors for which ground truth is provided in the `ann-benchmarks` datasets.
const NUM_NEIGHBORS: usize = 100;

/// A dataset from the `ann-benchmarks` repository.
pub struct AnnDataset<T: H5Type> {
    /// The data to use for clustering.
    pub train: Vec<Vec<T>>,
    /// The queries to use for search.
    pub queries: Vec<Vec<T>>,
    /// The true neighbors of each query, given as a tuple of:
    /// * index into `train`, and
    /// * distance to the query.
    pub neighbors: Vec<Vec<(usize, f32)>>,
}

/// Reads an `ann-benchmarks` dataset from the given path.
///
/// The dataset is expected be an HDF5 group with the following members:
///
/// * `train`: The data to use for clustering.
/// * `test`: The queries to use for search.
/// * `neighbors`: The true neighbors of each query, given as indices into `train`.
/// * `distances`: The distances from each query to its true neighbors.
///
/// If the dataset is flattened (as for `Kosarak` and `MovieLens10M`), then we expect
/// the following additional members:
///
/// * `size_train`: The lengths of each inner vector in `train`.
/// * `size_test`: The lengths of each inner vector in `test`.
///
/// # Arguments
///
/// * `path`: The path to the dataset.
/// * `flattened`: Whether to read the `train`
///
/// # Returns
///
/// The dataset, if it was read successfully.
///
/// # Errors
///
/// * If the path does not exist.
/// * If the path does not have the `.hdf5` extension.
/// * If the dataset is not readable.
/// * If the dataset is not in the expected format.
pub fn read<P: AsRef<Path>, T: H5Type>(path: &P, flattened: bool) -> Result<AnnDataset<T>, String> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    if !path.extension().map_or(false, |ext| ext == "hdf5") {
        return Err(format!("Path {path:?} does not have the `.hdf5` extension!"));
    }

    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    ftlog::info!("Opened file: {path:?}");

    let train_raw = file.dataset("train").map_err(|e| e.to_string())?;
    let test_raw = file.dataset("test").map_err(|e| e.to_string())?;
    ftlog::info!("Read raw train and test datasets.");

    let (train, queries) = if flattened {
        let train = train_raw.read_raw::<T>().map_err(|e| e.to_string())?;
        let size_train = file.dataset("size_train").map_err(|e| e.to_string())?;
        let size_train = size_train.read_raw::<usize>().map_err(|e| e.to_string())?;
        ftlog::info!("Reading {} flattened train items.", train.len());

        let queries = test_raw.read_raw::<T>().map_err(|e| e.to_string())?;
        let size_test = file.dataset("size_test").map_err(|e| e.to_string())?;
        let size_test = size_test.read_raw::<usize>().map_err(|e| e.to_string())?;
        ftlog::info!("Reading {} flattened test items.", queries.len());

        (
            abd_clam::utils::un_flatten(train, &size_train)?,
            abd_clam::utils::un_flatten(queries, &size_test)?,
        )
    } else {
        unimplemented!("Non-flattened datasets are not yet supported!")
    };
    ftlog::info!("Parsed {} train and {} query items.", train.len(), queries.len());

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
        ftlog::info!("Reading {} neighbors and distances.", neighbors.len());

        let neighbors = neighbors.into_iter().zip(distances).collect();
        let sizes = core::iter::repeat(NUM_NEIGHBORS)
            .take(queries.len())
            .collect::<Vec<_>>();
        abd_clam::utils::un_flatten(neighbors, &sizes)?
    };
    ftlog::info!("Parsed ground truth for {} queries.", neighbors.len());

    Ok(AnnDataset {
        train,
        queries,
        neighbors,
    })
}
