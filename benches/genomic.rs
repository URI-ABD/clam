use criterion::*;

use symagen::random_data;

use abd_clam::{
    cakes::{RnnAlgorithm, CAKES},
    cluster::PartitionCriteria,
    dataset::VecVec,
    COMMON_METRICS_STR,
};

fn genomic(c: &mut Criterion) {
    let mut group = c.benchmark_group("genomic".to_string());
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    let num_queries = 10;
    group.throughput(Throughput::Elements(num_queries as u64));

    println!("Generating data ...");
    let seed = 42;
    let cardinality = 1_000;
    let (min_len, max_len) = (100, 120);
    let alphabet = "ACGT";

    let data = random_data::random_string(cardinality, min_len, max_len, alphabet, seed);
    let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);

    println!("Building dataset ...");
    let data = data.iter().map(|v| v.as_str()).collect::<Vec<_>>();
    let queries = queries.iter().map(|v| v.as_str()).collect::<Vec<_>>();

    for &(metric_name, metric) in COMMON_METRICS_STR {
        println!("Building cakes for {} ...", metric_name);
        let name = format!("{}-{}", metric_name, cardinality);
        let dataset = VecVec::new(data.clone(), metric, name, true);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = CAKES::new(dataset, Some(seed), criteria);

        println!("Running benchmark for {} ...", metric_name);
        for radius in [10, 25, 50, 60] {
            let id = BenchmarkId::new(metric_name.to_string(), radius);
            group.bench_with_input(id, &radius, |b, &radius| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, RnnAlgorithm::Clustered));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, genomic);
criterion_main!(benches);
