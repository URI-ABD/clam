//! Reading from the hdf5 datasets.

use std::path::Path;

use crate::{member_set::MemberSet, CoSet, QueriesSet};

/// Reads a `Kosarak` dataset from the given path.
pub fn read<P: AsRef<Path>>(path: &P) -> Result<(CoSet, QueriesSet), String> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(format!("Path {path:?} does not exist!"));
    }

    if !path.extension().map_or(false, |ext| ext == "hdf5") {
        return Err(format!("Path {path:?} does not have the `.hdf5` extension!"));
    }

    let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
    let distances = file.dataset("distances").map_err(|e| e.to_string())?;
    let neighbors = file.dataset("neighbors").map_err(|e| e.to_string())?;
    let size_test = file.dataset("size_test").map_err(|e| e.to_string())?;
    let size_train = file.dataset("size_train").map_err(|e| e.to_string())?;
    let test = file.dataset("test").map_err(|e| e.to_string())?;
    let train = file.dataset("train").map_err(|e| e.to_string())?;

    // Convert the datasets into vectors.
    let _distances = distances.read_raw::<f32>().map_err(|e| e.to_string())?;
    let _neighbors = neighbors.read_raw::<usize>().map_err(|e| e.to_string())?;
    let size_test = size_test.read_raw::<usize>().map_err(|e| e.to_string())?;
    let size_train = size_train.read_raw::<usize>().map_err(|e| e.to_string())?;
    let test = test.read_raw::<usize>().map_err(|e| e.to_string())?;
    let train = train.read_raw::<usize>().map_err(|e| e.to_string())?;

    // The `size_test` and `size_train` vectors give the lengths of each inner vector
    // in the `test` and `train` vectors, respectively.
    let queries = {
        let mut test_iter = test.iter();
        let mut queries = Vec::with_capacity(size_test.len());
        for &s in &size_test {
            let mut query = Vec::with_capacity(s);
            for _ in 0..s {
                query.push(test_iter.next().copied().ok_or("Not enough elements in `test`!")?);
            }
            queries.push(MemberSet::new(&query));
        }
        queries
    };

    let data = {
        let mut train_iter = train.iter();
        let mut data = Vec::with_capacity(size_train.len());
        for &s in &size_train {
            let mut set = Vec::with_capacity(s);
            for _ in 0..s {
                set.push(train_iter.next().copied().ok_or("Not enough elements in `train`!")?);
            }
            data.push(MemberSet::new(&set));
        }
        data
    };

    // Get some statistics about the data and the queries.
    let data_lengths = data.iter().map(MemberSet::len).collect::<Vec<_>>();
    let (min_data_len, max_data_len) = data_lengths
        .iter()
        .fold((usize::MAX, 0), |(min, max), &len| (min.min(len), max.max(len)));
    let mean_data_len = abd_clam::utils::mean::<_, f32>(&data_lengths);
    let std_data_len = abd_clam::utils::standard_deviation::<_, f32>(&data_lengths);

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Data: Min len: {min_data_len}, Max len: {max_data_len}, Mean len: {mean_data_len}, Std len: {std_data_len}."
    );

    let queries_lengths = queries.iter().map(MemberSet::len).collect::<Vec<_>>();
    let (min_queries_len, max_queries_len) = queries_lengths
        .iter()
        .fold((usize::MAX, 0), |(min, max), &len| (min.min(len), max.max(len)));
    let mean_queries_len = abd_clam::utils::mean::<_, f32>(&queries_lengths);
    let std_queries_len = abd_clam::utils::standard_deviation::<_, f32>(&queries_lengths);

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Queries: Min len: {min_queries_len}, Max len: {max_queries_len}, Mean len: {mean_queries_len}, Std len: {std_queries_len}."
    );

    // TODO: There are duplicates in the data and queries. We have removed them when making the `MemberSet`s.
    // // Check if the data and the queries have the correct lengths.
    // if data_lengths != size_train {
    //     let pos = data_lengths.iter().zip(size_train.iter()).position(|(a, b)| a != b).unwrap_or_else(|| unreachable!("We know that the lengths are not equal!"));
    //     return Err(format!("Data length at position {pos} is {}, while it should be {}!", data_lengths[pos], size_train[pos]));
    // }

    // if queries_lengths != size_test {
    //     let pos = queries_lengths.iter().zip(size_test.iter()).position(|(a, b)| a != b).unwrap_or_else(|| unreachable!("We know that the lengths are not equal!"));
    //     return Err(format!("Queries length at position {pos} is {}, while it should be {}!", queries_lengths[pos], size_test[pos]));
    // }

    mt_logger::mt_log!(
        mt_logger::Level::Info,
        "Read {} sets from {path:?}. Minimum len: {min_data_len}, Maximum len: {max_data_len}.",
        data.len()
    );

    let metric = crate::metrics::SetDistance::Jaccard.metric();
    let data = CoSet::new(data, metric)?
        .with_dim_lower_bound(min_data_len)
        .with_dim_upper_bound(max_data_len);

    let queries = queries.into_iter().enumerate().collect();

    Ok((data, queries))
}
