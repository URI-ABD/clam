use std::time::Instant;

use num_format::{Locale, ToFormattedString};

use abd_clam::{cakes::CAKES, cluster::PartitionCriteria, dataset::VecVec};

pub mod utils;

use utils::search_readers;

fn main() {
    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS {
        // if !data_name.contains("glove") {
        //     continue;
        // }
        if data_name <= "gist" {
            continue;
        }
        if data_name == "nytimes" || data_name == "lastfm" {
            continue;
        }
        if metric_name == "jaccard" {
            continue;
        }
        println!();
        println!("Running knn-search on {data_name} ...");

        let metric = distances::vectors::euclidean_sq::<f32, f32>;

        let (data, queries) = search_readers::read_search_data(data_name).unwrap();
        let data = data.iter().map(|v| v.as_slice()).collect();
        let queries = queries.iter().map(|v| v.as_slice()).collect::<Vec<_>>();

        let data = VecVec::new(data, metric, data_name.to_string(), false);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);

        let start = Instant::now();
        let cakes = CAKES::new(data, Some(42)).build(criteria);
        let duration = start.elapsed().as_secs_f32();
        println!("Built the tree in {duration:.3} seconds ...");

        let num_queries = queries.len();

        for f in [10, 25, 50, 100].into_iter().rev() {
            // for k in [1, 10, 100] {
            println!("Running rnn-search on {data_name} with {num_queries} queries and factor = {f} ...");
            let radius = cakes.radius() / f as f32;

            let start = Instant::now();
            let results = cakes.par_batch_rnn_search(&queries, radius);
            let duration = start.elapsed().as_secs_f32();

            let num_results = results.into_iter().map(|v| v.len()).sum::<usize>();
            let num_results = num_results as f32 / num_queries as f32;

            let throughput = num_queries as f32 / duration;
            let t = (throughput as usize).to_formatted_string(&Locale::en);

            println!("Completed with throughput (queries per second) of {throughput:.2} ({t}) with mean result size {num_results:.3}");
            if num_results >= 500. {
                break;
            }
        }
    }
}
