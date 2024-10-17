//! Reading from the hdf5 datasets provided by the `ann-benchmarks` repository
//! on GitHub.

use super::AnnDataset;

/// `ann-benchmarks` datasets are expected to have this many neighbors.
const ANN_NUM_NEIGHBORS: usize = 100;

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
pub fn read<P: AsRef<std::path::Path>, T: hdf5::H5Type + Clone>(
    path: &P,
    flattened: bool,
) -> Result<AnnDataset<T>, String> {
    let path = path.as_ref();

    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    ftlog::info!("Opened file: {path:?}");

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

    Ok(AnnDataset {
        train,
        queries,
        neighbors,
    })
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
