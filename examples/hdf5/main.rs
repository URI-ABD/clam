mod h5data;
mod h5number;
mod h5space;

use clam::prelude::*;
use std::io::Write;

pub fn read_search_data(name: &str) -> hdf5::Result<hdf5::File> {
    let mut data_dir = std::env::current_dir().unwrap();
    data_dir.pop();
    data_dir.push("data");
    data_dir.push("search_data");
    data_dir.push("as_hdf5");

    let train_path = {
        let mut train_path = data_dir.clone();
        train_path.push(format!("{}.hdf5", name));
        assert!(train_path.exists(), "{:?} does not exist.", &train_path);
        train_path
    };

    hdf5::File::open(train_path)
}

#[derive(Debug, Clone)]
struct MinRadius<U: Number>(U);

impl<T: Number, U: Number> clam::criteria::PartitionCriterion<T, U> for MinRadius<U> {
    fn check(&self, c: &Cluster<T, U>) -> bool {
        c.radius() > self.0
    }
}

fn search<Tr, Te, T>(data_name: &str, metric_name: &str) -> Result<(), String>
where
    Tr: crate::h5number::H5Number, // For reading "train" data from hdf5 files.
    Te: crate::h5number::H5Number, // For reading "test" data from hdf5 files.
    T: clam::Number,               // For converting Tr and Te away from bool for kosarak.
{
    print!("{}, {}, ", data_name, metric_name);
    std::io::stdout().flush().unwrap();

    let file =
        read_search_data(data_name).map_err(|reason| format!("Could not read file {} because {}", data_name, reason))?;

    // let neighbors: Vec<Vec<usize>> = h5data::H5Data::new(&file, "neighbors", format!("{}_neighbors", data_name))?
    //     .to_vec_vec::<i32>()?
    //     .into_iter()
    //     .map(|row| row.into_iter().map(|v| v as usize).collect())
    //     .collect();

    // let distances = h5data::H5Data::new(&file, "distances", format!("{}_distances", data_name))?.to_vec_vec::<f32>()?;

    // let search_radii: Vec<f32> = distances
    //     .into_iter()
    //     .map(|row| clam::utils::helpers::argmax(&row).1)
    //     .collect();

    let queries = h5data::H5Data::new(&file, "test", format!("{}_test", data_name))?.to_vec_vec::<Te, T>()?;
    let num_queries = queries.len();
    print!("{}, ", num_queries);
    std::io::stdout().flush().unwrap();

    // let queries_radii: Vec<(Vec<T>, f32)> = queries.into_iter().zip(search_radii.into_iter()).collect();

    let train_vec = h5data::H5Data::new(&file, "train", "temp".to_string())?.to_vec_vec::<Tr, T>()?;
    let train = clam::Tabular::new(&train_vec, format!("{}_train", data_name));

    let metric = clam::metric_from_name::<T, f32>(metric_name)?;

    let space = clam::TabularSpace::new(&train, metric, false);

    // let log_cardinality = (space.data().cardinality() as f64).log2() as usize;
    // let partition_criteria =
    //     clam::criteria::PartitionCriteria::<T, f32>::new(true).with_min_cardinality(1 + log_cardinality);
    let cakes = clam::CAKES::new(&space);
    let search_radius = if metric_name == "euclidean" {
        cakes.radius() / 100.
    } else {
        0.01
    };
    let partition_criteria =
        clam::criteria::PartitionCriteria::<T, f32>::new(true).with_custom(Box::new(MinRadius(search_radius)));

    let start = std::time::Instant::now();
    let cakes = cakes.build(&partition_criteria);
    let build_time = start.elapsed().as_secs_f64();
    let tree_depth = cakes.depth();
    print!("{}, {:.2e}, ", tree_depth, build_time);
    std::io::stdout().flush().unwrap();

    let queries_radii: Vec<(Vec<T>, f32)> = queries.into_iter().map(|q| (q, search_radius)).collect();

    // let true_hits: Vec<HashSet<usize>> = neighbors.into_iter().map(|n_row| n_row.into_iter().collect()).collect();

    let num_runs = 1;
    let start = std::time::Instant::now();
    let hits: Vec<_> = (0..num_runs).map(|_| cakes.batch_rnn_search(&queries_radii)).collect();
    let end = start.elapsed().as_secs_f64();
    let search_time = end / (num_runs as f64);

    // let hits: Vec<HashSet<usize>> = hits
    //     .first()
    //     .unwrap()
    //     .into_iter()
    //     .map(|row| HashSet::from_iter(row.into_iter().map(|(v, _)| *v)))
    //     .collect();
    // let recalls: Vec<f64> = hits
    //     .into_iter()
    //     .zip(true_hits.into_iter())
    //     .map(|(pred, actual)| {
    //         let intersection = pred.intersection(&actual).count();
    //         (intersection as f64) / (actual.len() as f64)
    //     })
    //     .collect();

    // let mean_recall = clam::utils::helpers::mean(&recalls);

    let output_sizes: Vec<f64> = hits.first().unwrap().iter().map(|row| row.len() as f64).collect();
    let mean_output_size = clam::utils::helpers::argmax(&output_sizes).1;

    let time_per_query = search_time / (num_queries as f64);
    let queries_per_second = (num_queries as f64) / search_time;
    println!(
        "{:.2e}, {:.2e}, {:.5}",
        time_per_query, queries_per_second, mean_output_size
    );

    Ok(())
}

fn main() -> Result<(), String> {
    println!(
        "dataset, metric, num_queries, tree_depth, build_time (s), time_per_query (s), queries_per_second (1/s), mean_output_size"
    );

    // search::<f32, f32, f32>("deep-image", "cosine")?;
    search::<f32, f32, f32>("fashion-mnist", "euclidean")?;
    // search::<f32, f32, f32>("gist", "euclidean")?;
    // search::<f32, f32, f32>("glove-25", "cosine")?;
    // search::<f32, f32, f32>("glove-50", "cosine")?;
    // search::<f32, f32, f32>("glove-100", "cosine")?;
    // search::<f32, f32, f32>("glove-200", "cosine")?;
    // search::<bool, bool, f32>("kosarak", "jaccard")?;
    search::<f32, f32, f32>("mnist", "euclidean")?;
    // search::<f32, f32, f32>("nytimes", "cosine")?;
    // search::<f32, f32, f32>("sift", "euclidean")?;
    // search::<f32, f64, f32>("lastfm", "cosine")?;
    Ok(())
}
