//! Deal with the Radio-ML dataset.

use abd_clam::{dataset::DatasetIO, Dataset, FlatVec};
use bench_utils::Complex;
use rand::prelude::*;
use rayon::prelude::*;

/// Read the radio-ml dataset and subsamples it to a given maximum power of 2,
/// saving each subsample to a file.
pub fn read_and_subsample<P: AsRef<std::path::Path> + Send + Sync>(
    inp_dir: &P,
    out_dir: &P,
    n_queries: usize,
    max_power: u32,
    seed: Option<u64>,
    snr: Option<i32>,
) -> Result<(Vec<Vec<Complex<f64>>>, Vec<std::path::PathBuf>), String> {
    let name = "radio-ml"; // hard-coded until we add more datasets
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
        ftlog::info!("Reading radio-ml dataset from {:?}...", inp_dir.as_ref());
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
        let modulation_modes = bench_utils::radio_ml::ModulationMode::all();
        let (signals, queries) = {
            let signals = modulation_modes
                .par_iter()
                .map(|mode| bench_utils::radio_ml::read_mod(inp_dir, mode, snr))
                .collect::<Result<Vec<_>, _>>()?;
            let mut signals = signals.into_iter().flatten().collect::<Vec<_>>();
            signals.shuffle(&mut rng);
            let queries = signals.split_off(n_queries);
            (queries, signals)
        };
        ftlog::info!("Read {} signals and {} queries.", signals.len(), queries.len());

        ftlog::info!("Writing queries to {queries_path:?}...");
        let query_bytes = bitcode::encode(&queries).map_err(|e| e.to_string())?;
        std::fs::write(queries_path, query_bytes).map_err(|e| e.to_string())?;

        let dim = signals[0].len();
        let mut data = FlatVec::new(signals)?
            .with_name(name)
            .with_dim_lower_bound(dim)
            .with_dim_upper_bound(dim);

        ftlog::info!("Writing dataset to bitcode encoding: {data_path:?}");
        data.write_to(&data_path)?;

        for (power, path) in (1..=max_power).zip(all_paths.iter()) {
            let size = data.cardinality() / 2;
            ftlog::info!("Subsampling dataset with cardinality {size} to {path:?}...");
            data = data
                .random_subsample(&mut rng, size)
                .with_name(&format!("{name}-{power}"));
            ftlog::info!("Writing subsampled dataset with cardinality {size} to {path:?}...");
            data.write_to(path)?;
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
