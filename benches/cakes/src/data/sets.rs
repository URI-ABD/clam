//! Dealing with datasets of sets.

use abd_clam::{Dataset, FlatVec};
use rand::prelude::*;

/// Reads a dataset of sets from a file.
pub fn read<P: AsRef<std::path::Path>>(
    inp_dir: &P,
    data_name: &str,
    n_queries: usize,
    seed: Option<u64>,
) -> Result<(FlatVec<Vec<usize>, usize>, Vec<Vec<usize>>), String> {
    let inp_path = inp_dir.as_ref().join(format!("{data_name}.hdf5"));
    if !inp_path.exists() {
        return Err(format!("File not found: {}", inp_path.display()));
    }

    let data: bench_utils::ann_benchmarks::AnnDataset<usize> = bench_utils::ann_benchmarks::read(&inp_path, true)?;
    let (train, queries) = (data.train, data.queries);

    let lengths = train.iter().chain(queries.iter()).map(Vec::len).collect::<Vec<_>>();
    let (min_len, max_len, median, mean, std) = bench_utils::fasta::len_stats(&lengths);
    ftlog::info!(
        "Train set: {} sets, min_len: {min_len}, max_len: {max_len}, median: {median}, mean: {mean}, std: {std:.2}",
        train.len(),
    );

    let data = FlatVec::new(train)?
        .with_dim_lower_bound(min_len)
        .with_dim_upper_bound(max_len)
        .with_name(data_name);

    let queries = {
        let mut queries = queries;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(0));
        queries.shuffle(&mut rng);
        queries.truncate(n_queries);
        queries
    };

    Ok((data, queries))
}
