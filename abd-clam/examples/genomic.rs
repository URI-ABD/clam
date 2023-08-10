use std::time::Instant;

use symagen::random_data;

use abd_clam::{Cakes, PartitionCriteria, RnnAlgorithm, VecDataset, COMMON_METRICS_STR};

fn main() {
    let seed = 42;
    let (alphabet, min_len, max_len) = ("ACTG", 100, 120);
    let (train_size, num_queries) = (10_000, 100);

    let data = random_data::random_string(train_size, min_len, max_len, alphabet, seed);

    let metric = COMMON_METRICS_STR[1].1;
    let data = VecDataset::new("genomic".to_string(), data, metric, true);

    let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
    let model = Cakes::new(data, Some(42), criteria);

    let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);
    for radius in [10, 25, 50, 60] {
        let start = Instant::now();
        let results = model.batch_rnn_search(&queries, radius, rnn::Algorithm::Clustered);
        let duration = start.elapsed().as_secs_f32();

        let num_results = results.into_iter().map(|v| v.len()).sum::<usize>();
        let num_results = num_results as f32 / num_queries as f32;

        let throughput = num_queries as f32 / duration;

        println!("Completed with throughput of {throughput:.2} QPS with mean result size {num_results:.3}");
    }
}
