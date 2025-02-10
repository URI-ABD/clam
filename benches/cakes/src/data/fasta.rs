//! Dealing with FASTA files.

use abd_clam::{Dataset, ParDiskIO};
use rand::prelude::*;

/// Reads a fasta dataset and subsamples it to a given maximum power of 2,
/// saving each subsample to a file.
///
/// # Returns
///
/// A tuple containing:
///
/// - A vector of queries. Each query is a tuple of the ID and sequence.
/// - A vector of paths to the subsampled datasets, in ascending order of
///   cardinality.
pub fn read_and_subsample<P: AsRef<std::path::Path>>(
    inp_dir: &P,
    out_dir: &P,
    remove_gaps: bool,
    n_queries: usize,
    max_power: u32,
    seed: Option<u64>,
) -> Result<(Vec<(String, String)>, Vec<std::path::PathBuf>), String> {
    let name = "silva-SSU-Ref"; // hard-coded until we add more datasets
    let fasta_path = inp_dir.as_ref().join(format!("{name}.fasta"));
    let data_path = out_dir.as_ref().join(format!("{name}-0.flat_vec"));
    let queries_path = out_dir.as_ref().join(format!("{name}-queries.bin"));
    let mut all_paths = Vec::with_capacity(max_power as usize + 2);
    for power in 1..=max_power {
        all_paths.push(out_dir.as_ref().join(format!("{name}-{power}.flat_vec")));
    }
    all_paths.push(data_path.clone());
    all_paths.push(queries_path.clone());

    let queries = if all_paths.iter().all(|p| p.exists()) {
        ftlog::info!("Subsampled datasets already exist. Reading queries from {queries_path:?}...");
        let bytes = std::fs::read(queries_path).map_err(|e| e.to_string())?;
        bitcode::decode(&bytes).map_err(|e| e.to_string())?
    } else {
        if !fasta_path.exists() {
            return Err(format!("Dataset {name} not found: {fasta_path:?}"));
        }

        ftlog::info!("Reading fasta dataset from fasta file: {fasta_path:?}");
        let (mut data, queries) = bench_utils::fasta::read(&fasta_path, n_queries, remove_gaps)?;
        data = data.with_name(name);

        ftlog::info!("Writing dataset to bitcode encoding: {data_path:?}");
        data.par_write_to(&data_path)?;

        let query_bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
        std::fs::write(queries_path, query_bytes).map_err(|e| e.to_string())?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
        for (power, path) in (1..=max_power).zip(all_paths.iter()) {
            let size = data.cardinality() / 2;
            ftlog::info!("Subsampling dataset with cardinality {size} to {path:?}...");
            data = data
                .random_subsample(&mut rng, size)
                .with_name(&format!("{name}-{power}"));
            ftlog::info!("Writing subsampled dataset with cardinality {size} to {path:?}...");
            data.par_write_to(path)?;
        }

        queries
    };

    all_paths.pop(); // remove queries_path
    let full_path = all_paths
        .pop()
        .unwrap_or_else(|| unreachable!("We added the path ourselves.")); // remove data_path
    all_paths.insert(0, full_path);
    all_paths.reverse(); // return in ascending order of cardinality

    Ok((queries, all_paths))
}
