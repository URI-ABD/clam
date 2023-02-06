use std::collections::HashSet;

// use rayon::prelude::*;

use clam::prelude::*;
use clam::utils::helpers;
// use clam::utils::reports;

mod h5data;
mod h5number;
mod h5space;
// mod utils;

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

fn compute_recall(hits: &[Vec<usize>], true_hits: &[HashSet<usize>]) -> Vec<f64> {
    let hits: Vec<HashSet<usize>> = hits.iter().map(|r| HashSet::from_iter(r.iter().copied())).collect();
    let recalls: Vec<f64> = hits
        .iter()
        .zip(true_hits)
        .map(|(pred, actual)| {
            let intersection = pred.intersection(actual).count();
            (intersection as f64) / (actual.len() as f64)
        })
        .collect();
    recalls
}

#[allow(clippy::single_element_loop)]
fn search<Tr, Te, N, T>(data_name: &str, metric_name: &str, num_runs: usize) -> Result<(), String>
where
    Tr: h5number::H5Number, // For reading "train" data from hdf5 files.
    Te: h5number::H5Number, // For reading "test" data from hdf5 files.
    N: h5number::H5Number,  // For reading ground-truth neighbors' indices.
    T: Number,              // For converting Tr and Te away from bool for kosarak data.
{
    log::info!("");
    log::info!("Running {}-{} data ...", data_name, metric_name);

    let file =
        open_hdf5_file(data_name).map_err(|reason| format!("Could not read file {data_name} because {reason}"))?;

    let neighbors =
        h5data::H5Data::<N>::new(&file, "neighbors", format!("{data_name}_neighbors"))?.to_vec_vec::<usize>()?;

    let distances =
        h5data::H5Data::<f64>::new(&file, "distances", format!("{data_name}_distances"))?.to_vec_vec::<f64>()?;

    let search_radii = distances
        .into_iter()
        .map(|row| clam::utils::helpers::arg_max(&row).1)
        .collect::<Vec<_>>();

    // Adding a 10% buffer to search radius to test change in recall
    // let search_radii: Vec<_> = distances
    //     .into_iter()
    //     .map(|row| clam::utils::helpers::arg_max(&row).1)
    //     .map(|v| D::from(v.as_f64() * 1.1).unwrap())
    //     .collect();

    let min_radius = clam::utils::helpers::arg_min(&search_radii).1;

    let queries = h5data::H5Data::<Te>::new(&file, "test", format!("{data_name}_test"))?.to_vec_vec::<T>()?;
    let queries = clam::Tabular::new(&queries, format!("{data_name}-queries"));
    // let num_queries = queries.cardinality();
    let num_queries = 1000;
    let queries = (0..num_queries).map(|i| queries.get(i)).collect::<Vec<_>>();
    // let queries = vec![queries.get(42)];
    // let num_queries = queries.len();

    let metric = clam::metric_from_name::<T>(metric_name, false)?;

    let train = h5data::H5Data::<Tr>::new(&file, "train", "temp".to_string())?;

    // because reading h5 data with this crate is too slow ...
    // let space = h5space::H5Space::new(&train, metric.as_ref(), false);
    let train = train.to_vec_vec::<T>()?;
    let train = clam::Tabular::new(&train, data_name.to_string());
    let space = clam::TabularSpace::new(&train, metric.as_ref());

    let partition_criteria = clam::PartitionCriteria::new(true)
        .with_min_cardinality(1)
        .with_custom(Box::new(MinRadius {
            threshold: min_radius.as_f64() / 1000.,
        }));

    log::info!("Building search tree on {}-{} data ...", data_name, metric_name);

    let start = std::time::Instant::now();
    let cakes = clam::CAKES::new(&space).build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();
    log::info!(
        "Built tree to a depth of {} in {:.2e} seconds ...",
        cakes.depth(),
        build_time
    );

    log::info!(
        "Starting knn-search on {}-{} data with {} queries ...",
        data_name,
        metric_name,
        num_queries
    );
    log::info!("");

    // fashion-mnist search times (ms per query) for 1_000 queries:
    // (before):          36.0, 48.3, 62.5
    // (after, full):     25.0, 35.0, 49.0

    // (sorting) multi-threaded search times (ms per query) for 1_000 queries
    // deep-image     , 105.          , 307.          , 387.
    // fashion-mnist  ,   4.58        ,   6.78        ,   8.77
    // gist           , 285.          , 297.          , 305.
    // glove-25       ,   8.81        ,  17.3         ,  22.7
    // glove-50       ,  51.4         ,  74.6         ,  96.4
    // glove-100      , 132.          , 156.          , 165.
    // glove-200      , 200.          , 212.          , 217.
    // lastfm         ,    .000161    ,    .000980    ,  31.9
    // mnist          ,   8.51        ,  10.9         ,  13.1
    // nytimes        ,  34.6         ,  41.3         ,  41.2
    // sift           ,  45.1         ,  57.5         ,  67.5

    // (find_kth) multi-threaded search times (ms per query) for 1_000 queries
    // deep-image     ,  85.3         , 300.          , 355.          //
    // fashion-mnist  ,   3.56        ,   4.78        ,   6.70        //
    // gist           , 257.          , 267.          , 274.          //
    // glove-25       ,   3.89        ,  10.7         ,  14.7         //
    // glove-50       ,  37.7         ,  53.6         ,  72.2         //
    // glove-100      , 121.          , 129.          , 139.          //
    // glove-200      , 184.          , 192.          , 193.          //
    // lastfm         ,    .000655    ,    .00153     ,  34.7         //
    // mnist          ,   8.21        ,  10.2         ,  12.0         //
    // nytimes        ,    .          ,    .          ,    .          // Stack-overflow error from recursion in find_kth. Tree was 254 deep.
    // sift           ,  37.3         ,  43.2         ,  52.0         //
    for k in [1, 10, 100] {
        // for k in [100] {
        log::info!("Using k = {} ...", k);
        log::info!("");

        let start = std::time::Instant::now();
        let knn_hits = (0..num_runs)
            .map(|_| cakes.batch_knn_by_rnn(&queries, k))
            .last()
            .unwrap();
        let time = start.elapsed().as_secs_f64() / (num_runs as f64);
        let mean_time = time / (num_queries as f64);

        let knn_hits = knn_hits
            .into_iter()
            .map(|hits| hits.into_iter().map(|(i, _)| i).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        log::info!("knn-search time: {:.2e} seconds per query ...", mean_time);
        log::info!("");

        if k == neighbors[0].len() {
            let true_hits = neighbors
                .iter()
                .map(|row| row.iter().copied().collect())
                .collect::<Vec<_>>();

            let recalls = compute_recall(&knn_hits, &true_hits);

            let mean_recall = helpers::mean(&recalls);
            let sd_recall = helpers::sd(&recalls, mean_recall);
            log::info!("knn-recall: {:.2e} +/- {:.2e} ...", mean_recall, sd_recall);
            log::info!("");
        }
    }
    log::info!("Moving on ...");
    log::info!("");

    Ok(())
}

fn main() -> Result<(), String> {
    env_logger::Builder::new().parse_filters("info").init();

    let results = [
        // search::<f32, f32, i32, f32>("deep-image", "cosine", 1),
        // search::<f32, f32, i32, f32>("fashion-mnist", "euclidean", 1),
        // search::<f32, f32, i32, f32>("gist", "euclidean", 1),
        // search::<f32, f32, i32, f32>("glove-25", "cosine", 1),
        // search::<f32, f32, i32, f32>("glove-50", "cosine", 1),
        // search::<f32, f32, i32, f32>("glove-100", "cosine", 1),
        // search::<f32, f32, i32, f32>("glove-200", "cosine", 1),
        // search::<f32, f64, i32, f32>("lastfm", "cosine", 1),
        // search::<f32, f32, i32, f32>("mnist", "euclidean", 1),
        search::<f32, f32, i32, f32>("nytimes", "cosine", 1),
        // search::<f32, f32, i32, f32>("sift", "euclidean", 1),
        // search::<bool, bool, i32, u8>("kosarak", "jaccard", 1),
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
