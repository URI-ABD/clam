use clam::prelude::*;

mod h5data;
mod h5number;
mod h5space;
mod reports;

pub fn open_hdf5_file(name: &str) -> hdf5::Result<hdf5::File> {
    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    data_dir.push("search_small");
    data_dir.push("as_hdf5");

    let train_path = {
        let mut train_path = data_dir.clone();
        train_path.push(format!("{name}.hdf5"));
        assert!(train_path.exists(), "{train_path:?} does not exist.");
        train_path
    };

    hdf5::File::open(train_path)
}

// An example for a custom partition criterion.
#[derive(Debug, Clone)]
struct MinRadius {
    threshold: f64,
}

impl<'a, T, S> clam::PartitionCriterion<'a, T, S> for MinRadius
where
    T: Number + 'a,
    S: Space<'a, T> + 'a,
{
    fn check(&self, c: &Cluster<'a, T, S>) -> bool {
        c.radius() > self.threshold
    }
}

fn search<Tr, Te, N, D, T>(data_name: &str, metric_name: &str, _num_runs: usize) -> Result<(), String>
where
    Tr: h5number::H5Number, // For reading "train" data from hdf5 files.
    Te: h5number::H5Number, // For reading "test" data from hdf5 files.
    N: h5number::H5Number,  // For reading ground-truth neighbors' indices.
    D: h5number::H5Number,  // For reading ground-truth neighbors' distances.
    T: Number,              // For converting Tr and Te away from bool for kosarak data.
{
    let run_name = format!("{data_name}__{metric_name}");
    let output_dir = {
        let mut output_dir = std::env::current_dir().unwrap();
        output_dir.pop();
        output_dir.push("data");
        output_dir.push("reports");
        assert!(output_dir.exists(), "Path not found: {output_dir:?}");
        output_dir.push("geometry");
        if !output_dir.exists() {
            std::fs::create_dir(&output_dir).unwrap();
        }
        output_dir.push(&run_name);
        if output_dir.exists() {
            std::fs::remove_dir_all(&output_dir).unwrap();
        }
        std::fs::create_dir(&output_dir).unwrap();
        output_dir
    };

    log::info!("Running {run_name} data ...");

    let file =
        open_hdf5_file(data_name).map_err(|reason| format!("Could not read file {data_name} because {reason}"))?;

    // let neighbors =
    //     h5data::H5Data::<N>::new(&file, "neighbors", format!("{}_neighbors", data_name))?.to_vec_vec::<usize>()?;

    // let distances =
    //     h5data::H5Data::<D>::new(&file, "distances", format!("{}_distances", data_name))?.to_vec_vec::<D>()?;

    // let search_radii: Vec<_> = distances
    //     .into_iter()
    //     .map(|row| clam::utils::helpers::arg_max(&row).1)
    //     .collect();

    // Adding a 10% buffer to search radius to test change in recall
    // let search_radii: Vec<_> = distances
    //     .into_iter()
    //     .map(|row| clam::utils::helpers::arg_max(&row).1)
    //     .map(|v| D::from(v.as_f64() * 1.1).unwrap())
    //     .collect();

    // let min_radius = clam::utils::helpers::arg_min(&search_radii).1;

    // let queries = h5data::H5Data::<Te>::new(&file, "test", format!("{}_test", data_name))?.to_vec_vec::<T>()?;

    // let queries_radii: Vec<(Vec<T>, D)> = queries.into_iter().zip(search_radii.iter().cloned()).collect();

    let metric = clam::metric_from_name::<T>(metric_name, false)?;

    let train = h5data::H5Data::<Tr>::new(&file, "train", "temp".to_string())?;

    // because reading h5 data with this crate is too slow ...
    // let space = h5space::H5Space::new(&train, metric.as_ref(), false);
    let train = train.to_vec_vec::<T>()?;
    let train = clam::Tabular::new(&train, data_name.to_string());
    let space = clam::TabularSpace::new(&train, metric.as_ref());

    let log_cardinality = (space.data().cardinality() as f64).log2() as usize;
    // let partition_criteria = clam::PartitionCriteria::new(true)
    //     .with_min_cardinality(10)
    //     .with_custom(Box::new(MinRadius {
    //         threshold: D::from(min_radius.as_f64() / 1000.).unwrap(),
    //     }));

    // let partition_criteria = clam::PartitionCriteria::new(true).with_max_depth(20);
    let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(log_cardinality);

    log::info!("Building search tree on {run_name} data ...");

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space);
    let cakes = cakes.build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();

    log::info!(
        "Built tree to a depth of {} in {build_time:.2e} seconds ...",
        cakes.depth()
    );
    log::info!("Writing tree report on {run_name} data ...");
    reports::report_tree(&output_dir, cakes.root(), build_time)?;

    //     log::info!("Starting search on {}-{} data ...", data_name, metric_name);

    //     let (hits, search_times): (Vec<_>, Vec<_>) = queries_radii
    //         .iter()
    //         .enumerate()
    //         .map(|(i, (query, radius))| {
    //             if (i + 1) % 10 == 0 {
    //                 log::info!(
    //                     "Progress {:6.2}% on {}-{} data ...",
    //                     100. * (i as f64 + 1.) / queries_radii.len() as f64,
    //                     data_name,
    //                     metric_name
    //                 );
    //             }
    //             let sample = (0..num_runs)
    //                 .map(|_| {
    //                     let start = std::time::Instant::now();
    //                     let results = cakes.rnn_search(query, *radius);
    //                     (results, start.elapsed().as_secs_f64())
    //                 })
    //                 .collect::<Vec<_>>();
    //             // let hits = sample.first().unwrap().0.clone();
    //             let hits = {
    //                 let mut hits = sample.first().unwrap().0.clone();
    //                 hits.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
    //                 if hits.len() > 100 {
    //                     let threshold = hits[100].1;
    //                     hits.into_iter().filter(|(_, d)| *d <= threshold).collect()
    //                 } else {
    //                     hits
    //                 }
    //             };
    //             let times = sample.into_iter().map(|(_, t)| t).collect();
    //             (hits, times)
    //         })
    //         .unzip();

    //     log::info!("Collecting report on {}-{} data ...", data_name, metric_name);

    //     let true_hits = neighbors.into_iter().map(|n_row| n_row.into_iter().collect());
    //     let hits: Vec<HashSet<usize>> = hits
    //         .iter()
    //         .map(|row| HashSet::from_iter(row.iter().map(|(v, _)| *v)))
    //         .collect();
    //     let recalls: Vec<f64> = hits
    //         .iter()
    //         .zip(true_hits)
    //         .map(|(pred, actual)| {
    //             let intersection = pred.intersection(&actual).count();
    //             (intersection as f64) / (actual.len() as f64)
    //         })
    //         .collect();

    //     let outputs = hits.iter().map(|row| row.iter().cloned().collect::<Vec<_>>()).collect();

    //     let report = reports::RnnReport {
    //         data_name,
    //         metric_name,
    //         num_queries: queries_radii.len(),
    //         num_runs,
    //         cardinality: train.cardinality(),
    //         dimensionality: train.dimensionality(),
    //         tree_depth: cakes.depth(),
    //         build_time,
    //         root_radius: cakes.radius().as_f64(),
    //         search_radii: search_radii.into_iter().map(|v| v.as_f64()).collect(),
    //         search_times,
    //         outputs,
    //         recalls,
    //     };

    //     let failures = report.is_valid();
    //     if failures.is_empty() {
    //         let report = serde_json::to_string_pretty(&report)
    //             .map_err(|reason| format!("Could not convert report to json because {}", reason))?;
    //         let output_path = output_dir.join(format!("{}-{}.json", data_name, metric_name));
    //         let mut file = std::fs::File::create(&output_path)
    //             .map_err(|reason| format!("Could not create/open file {:?} because {}", output_path, reason))?;
    //         Ok(write!(&mut file, "{}", report)
    //             .map_err(|reason| format!("Could not write report to {:?} because {}.", output_path, reason))?)
    //     } else {
    //         Err(format!("The report was invalid:\n{:?}", failures.join("\n")))
    //     }
    Ok(())
}

fn main() -> Result<(), String> {
    env_logger::Builder::new().parse_filters("info").init();

    let results = [
        // search::<f32, f32, i32, f32, f32>("deep-image", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("fashion-mnist", "euclidean", 10),
        search::<f32, f32, i32, f32, f32>("gist", "euclidean", 10),
        search::<f32, f32, i32, f32, f32>("glove-25", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-50", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-100", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("glove-200", "cosine", 10),
        // search::<f32, f64, i32, f32, f32>("lastfm", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("mnist", "euclidean", 10),
        // search::<f32, f32, i32, f32, f32>("nytimes", "cosine", 10),
        search::<f32, f32, i32, f32, f32>("sift", "euclidean", 10),
        // search::<bool, bool, i32, f32, u8>("kosarak", "jaccard", 10),
    ];
    println!(
        "Successful for {}/{} datasets.",
        results.iter().filter(|v| v.is_ok()).count(),
        results.len()
    );
    let failures: Vec<_> = results.iter().filter(|v| v.is_err()).cloned().collect();
    if !failures.is_empty() {
        println!("Failed for {}/{} datasets.", failures.len(), results.len());
        failures.into_iter().for_each(|v| println!("{v:?}\n"));
    }
    Ok(())
}
