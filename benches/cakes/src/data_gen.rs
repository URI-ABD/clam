//! Utilities for generating and reading datasets and ground-truth search results.

use abd_clam::{
    dataset::{DatasetIO, ParDataset},
    metric::ParMetric,
    Dataset, FlatVec,
};
use bench_utils::{ann_benchmarks::AnnDataset, Complex};
use distances::Number;
use rand::prelude::*;
use rayon::prelude::*;

/// Read the radio-ml dataset and subsamples it to a given maximum power of 2,
/// saving each subsample to a file.
#[allow(clippy::type_complexity)]
pub fn read_radio_ml_and_subsample<P: AsRef<std::path::Path> + Send + Sync>(
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

/// Reads a fasta dataset and subsamples it to a given maximum power of 2,
/// saving each subsample to a file.
#[allow(clippy::type_complexity)]
pub fn read_fasta_and_subsample<P: AsRef<std::path::Path>>(
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
        data.write_to(&data_path)?;

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

/// Read the tabular floating-point dataset and augment it to a given maximum
/// power of 2.
///
/// # Arguments
///
/// - `dataset`: The dataset to read.
/// - `metric`: The metric, if provided, to use for linear search to find true
///   neighbors of augmented datasets.
/// - `max_power`: The maximum power of 2 to which the cardinality of the
///   dataset should be augmented.
/// - `seed`: The seed for the random number generator.
/// - `inp_dir`: The directory containing the input dataset.
/// - `out_dir`: The directory to which the augmented datasets and ground-truth
///   neighbors and distances should be written.
///
/// # Returns
///
/// The queries to use for benchmarking.
///
/// # Errors
///
/// - If there is an error reading the input dataset.
/// - If there is an error writing the augmented datasets.
#[allow(clippy::too_many_lines)]
pub fn read_tabular_and_augment<P: AsRef<std::path::Path>, M: ParMetric<Vec<f32>, f32>>(
    dataset: &bench_utils::RawData,
    metric: Option<&M>,
    max_power: u32,
    seed: Option<u64>,
    inp_dir: &P,
    out_dir: &P,
) -> Result<Vec<Vec<f32>>, String> {
    let ([data_orig_path, queries_path, neighbors_path, distances_path], all_paths) =
        gen_all_paths(dataset, max_power, out_dir);

    if all_paths.iter().all(|p| p.exists()) {
        ftlog::info!("Augmented datasets already exist. Reading queries from {queries_path:?}...");
        return FlatVec::<Vec<f32>, usize>::read_npy(&queries_path).map(FlatVec::take_items);
    }

    ftlog::info!("Reading data from {:?}...", inp_dir.as_ref());
    let data = dataset.read_vector::<_, f32>(&inp_dir)?;
    let (train, queries, neighbors) = (data.train, data.queries, data.neighbors);
    let (neighbors, distances): (Vec<_>, Vec<_>) = neighbors
        .into_iter()
        .map(|n| {
            let (n, d): (Vec<_>, Vec<_>) = n.into_iter().unzip();
            let n = n.into_iter().map(Number::as_u64).collect::<Vec<_>>();
            (n, d)
        })
        .unzip();
    let k = neighbors[0].len();
    let neighbors = FlatVec::new(neighbors)?.with_dim_lower_bound(k).with_dim_upper_bound(k);
    let distances = FlatVec::new(distances)?.with_dim_lower_bound(k).with_dim_upper_bound(k);

    let (min_dim, max_dim) = train
        .iter()
        .chain(queries.iter())
        .fold((usize::MAX, 0), |(min, max), x| {
            (Ord::min(min, x.len()), Ord::max(max, x.len()))
        });

    let data = FlatVec::new(train)?
        .with_name(dataset.name())
        .with_dim_lower_bound(min_dim)
        .with_dim_upper_bound(max_dim);

    ftlog::info!("Writing original data as npy array to {data_orig_path:?}...");
    data.write_npy(&data_orig_path)?;

    let query_data = FlatVec::new(queries)?
        .with_name(&format!("{}-queries", dataset.name()))
        .with_dim_lower_bound(min_dim)
        .with_dim_upper_bound(max_dim);
    ftlog::info!("Writing queries as npy array to {queries_path:?}...");
    query_data.write_npy(&queries_path)?;
    let queries = query_data.take_items();

    ftlog::info!("Writing neighbors to {neighbors_path:?}...");
    neighbors.write_npy(&neighbors_path)?;
    distances.write_npy(&distances_path)?;

    ftlog::info!("Augmenting data...");
    let train = data.take_items();
    let base_cardinality = train.len();
    let data = AnnDataset {
        train,
        queries: Vec::new(),
        neighbors: Vec::new(),
    }
    .augment(1 << max_power, 0.1);

    // The value of k is hardcoded to 100 to find the true neighbors of the
    // augmented datasets.
    let k = 100;

    let mut data = FlatVec::new(data.train)?
        .with_dim_lower_bound(min_dim)
        .with_dim_upper_bound(max_dim);

    for power in (1..=max_power).rev() {
        let name = format!("{}-{power}", dataset.name());
        let data_path = out_dir.as_ref().join(format!("{name}.npy"));
        let neighbors_path = out_dir.as_ref().join(format!("{name}-neighbors.npy"));
        let distances_path = out_dir.as_ref().join(format!("{name}-distances.npy"));

        let size = base_cardinality * (1 << power);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
        data = data.random_subsample(&mut rng, size).with_name(&name);
        ftlog::info!("Writing {}x augmented data to {data_path:?}...", 1 << power);
        data.write_npy(&data_path)?;

        if let Some(metric) = metric {
            ftlog::info!("Finding true neighbors for {name}...");
            let indices = (0..data.cardinality()).collect::<Vec<_>>();
            let true_hits = queries
                .par_iter()
                .map(|query| {
                    let mut hits = data.par_query_to_many(query, &indices, metric).collect::<Vec<_>>();
                    hits.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                    let _ = hits.split_off(k);
                    hits
                })
                .collect::<Vec<_>>();
            ftlog::info!("Writing true neighbors to {neighbors_path:?} and distances to {distances_path:?}...");
            let (neighbors, distances): (Vec<_>, Vec<_>) = true_hits
                .into_iter()
                .map(|mut nds| {
                    // Sort the neighbors by distance from the query.
                    nds.sort_by(|(_, a), (_, b)| a.total_cmp(b));

                    let (n, d): (Vec<_>, Vec<_>) = nds.into_iter().unzip();
                    let n = n.into_iter().map(Number::as_u64).collect::<Vec<_>>();

                    (n, d)
                })
                .unzip();
            FlatVec::new(neighbors)?
                .with_dim_lower_bound(k)
                .with_dim_upper_bound(k)
                .write_npy(&neighbors_path)?;
            FlatVec::new(distances)?
                .with_dim_lower_bound(k)
                .with_dim_upper_bound(k)
                .write_npy(&distances_path)?;
        }
    }

    Ok(queries)
}

/// Read or generate random datasets and ground-truth search results.
///
/// # Arguments
///
/// - `metric`: The metric, if provided, to use for linear search to find true
///   neighbors of augmented datasets.
/// - `max_power`: The maximum power of 2 to which the cardinality of the
///   dataset should be augmented.
/// - `seed`: The seed for the random number generator.
/// - `out_dir`: The directory to which the augmented datasets and ground-truth
///   neighbors and distances should be written.
///
/// # Returns
///
/// The queries to use for benchmarking.
pub fn read_or_gen_random<P: AsRef<std::path::Path>, M: ParMetric<Vec<f32>, f32>>(
    metric: Option<&M>,
    max_power: u32,
    seed: Option<u64>,
    out_dir: &P,
) -> Result<Vec<Vec<f32>>, String> {
    let dataset = bench_utils::RawData::Random;
    let ([_, queries_path, _, _], all_paths) = gen_all_paths(&dataset, max_power, out_dir);

    if all_paths.iter().all(|p| p.exists()) {
        ftlog::info!("Random datasets already exist. Reading queries from {queries_path:?}...");
        return FlatVec::<Vec<f32>, usize>::read_npy(&queries_path).map(FlatVec::take_items);
    }
    let k = 100;
    let n_queries = 100;
    let base_cardinality = 1_000_000;
    let dimensionality = 128;
    let data = AnnDataset::gen_random(base_cardinality, 1 << max_power, dimensionality, n_queries, 42);
    let (train, queries, _) = (data.train, data.queries, data.neighbors);

    let queries = FlatVec::new(queries)?
        .with_dim_lower_bound(dimensionality)
        .with_dim_upper_bound(dimensionality);
    let queries_path = out_dir.as_ref().join(format!("{}-queries.npy", dataset.name()));
    ftlog::info!("Writing queries as npy array to {queries_path:?}...");
    queries.write_npy(&queries_path)?;
    let queries = queries.take_items();

    let mut data = FlatVec::new(train)?
        .with_dim_lower_bound(dimensionality)
        .with_dim_upper_bound(dimensionality);
    for power in (0..=max_power).rev() {
        let name = format!("{}-{}", dataset.name(), power);
        let data_path = out_dir.as_ref().join(format!("{name}.npy"));
        let neighbors_path = out_dir.as_ref().join(format!("{name}-neighbors.npy"));
        let distances_path = out_dir.as_ref().join(format!("{name}-distances.npy"));

        let size = base_cardinality * (1 << power);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed.unwrap_or(42));
        data = data.random_subsample(&mut rng, size).with_name(&name);
        ftlog::info!("Writing {}x random data to {data_path:?}...", 1 << power);
        data.write_npy(&data_path)?;

        if let Some(metric) = metric {
            ftlog::info!("Finding true neighbors for {name}...");
            let indices = (0..data.cardinality()).collect::<Vec<_>>();
            let true_hits = queries
                .par_iter()
                .map(|query| {
                    let mut hits = data.par_query_to_many(query, &indices, metric).collect::<Vec<_>>();
                    hits.sort_by(|(_, a), (_, b)| a.total_cmp(b));
                    let _ = hits.split_off(k);
                    hits
                })
                .collect::<Vec<_>>();
            ftlog::info!("Writing true neighbors to {neighbors_path:?} and distances to {distances_path:?}...");
            let (neighbors, distances): (Vec<_>, Vec<_>) = true_hits
                .into_iter()
                .map(|mut nds| {
                    // Sort the neighbors by distance from the query.
                    nds.sort_by(|(_, a), (_, b)| a.total_cmp(b));

                    let (n, d): (Vec<_>, Vec<_>) = nds.into_iter().unzip();
                    let n = n.into_iter().map(Number::as_u64).collect::<Vec<_>>();

                    (n, d)
                })
                .unzip();
            FlatVec::new(neighbors)?
                .with_dim_lower_bound(k)
                .with_dim_upper_bound(k)
                .write_npy(&neighbors_path)?;
            FlatVec::new(distances)?
                .with_dim_lower_bound(k)
                .with_dim_upper_bound(k)
                .write_npy(&distances_path)?;
        }
    }

    Ok(queries)
}

/// Generate all paths for the augmented datasets and ground-truth neighbors and
/// distances.
fn gen_all_paths<P: AsRef<std::path::Path>>(
    dataset: &bench_utils::RawData,
    max_power: u32,
    out_dir: &P,
) -> ([std::path::PathBuf; 4], Vec<std::path::PathBuf>) {
    let data_orig_path = out_dir.as_ref().join(format!("{}-0.npy", dataset.name()));
    let queries_path = out_dir.as_ref().join(format!("{}-queries.npy", dataset.name()));
    let neighbors_path = out_dir.as_ref().join(format!("{}-0-neighbors.npy", dataset.name()));
    let distances_path = out_dir.as_ref().join(format!("{}-0-distances.npy", dataset.name()));

    let all_paths = {
        let mut paths = Vec::with_capacity(max_power as usize + 3);
        for power in 1..=max_power {
            paths.push(out_dir.as_ref().join(format!("{}-{}.npy", dataset.name(), power)));
            paths.push(
                out_dir
                    .as_ref()
                    .join(format!("{}-{}-neighbors.npy", dataset.name(), power)),
            );
            paths.push(
                out_dir
                    .as_ref()
                    .join(format!("{}-{}-distances.npy", dataset.name(), power)),
            );
        }
        paths.push(data_orig_path.clone());
        paths.push(queries_path.clone());
        paths
    };

    (
        [data_orig_path, queries_path, neighbors_path, distances_path],
        all_paths,
    )
}
