use std::time::Instant;

use distances::strings::levenshtein;
use symagen::random_data;

use abd_clam::{PartitionCriteria, RnnAlgorithm, VecDataset, CAKES};

fn main() {
    let seed = 42;
    let (alphabet, min_len, max_len) = ("ACTG", 100, 120);
    let (train_size, num_queries) = (10_000, 100);

    let data = random_data::random_string(train_size, min_len, max_len, alphabet, seed);
    let data = data.iter().map(|v| v.as_str()).collect();

    let metric = levenshtein::<u16>;
    let data = VecDataset::new(data, metric, "genomic".to_string(), true);

    let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
    let model = CAKES::new(data, Some(42), criteria);

    let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);
    let queries = queries.iter().map(|v| v.as_str()).collect::<Vec<_>>();
    for radius in [10, 25, 50, 60] {
        let start = Instant::now();
        let results = model.batch_rnn_search(&queries, radius, RnnAlgorithm::Clustered);
        let duration = start.elapsed().as_secs_f32();

        let num_results = results.into_iter().map(|v| v.len()).sum::<usize>();
        let num_results = num_results as f32 / num_queries as f32;

        let throughput = num_queries as f32 / duration;

        println!("Completed with throughput of {throughput:.2} QPS with mean result size {num_results:.3}");
    }
}
