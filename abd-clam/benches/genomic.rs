use criterion::*;

use symagen::random_data;

use abd_clam::{rnn, Cakes, PartitionCriteria, VecDataset, COMMON_METRICS_STR};

fn genomic(c: &mut Criterion) {
    let seed = 42;
    let cardinality = 10_000;
    let min_len = 1000;
    let max_len = 1000;
    let alphabet = "ACGT";
    let num_queries = 10;

    println!("Building dataset ...");
    let data = random_data::random_string(cardinality, min_len, max_len, alphabet, seed);
    let data = data.iter().map(String::as_str).collect::<Vec<_>>();

    let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);
    let queries = queries.iter().map(String::as_str).collect::<Vec<_>>();

    for &(metric_name, metric) in &COMMON_METRICS_STR[..2] {
        let mut group = c.benchmark_group(format!("genomic-{metric_name}"));
        group
            .sampling_mode(SamplingMode::Flat)
            .throughput(Throughput::Elements(num_queries as u64))
            .plot_config(PlotConfiguration::default().summary_scale(AxisScale::Linear));

        println!("Building cakes for {metric_name} ...");
        let data_name = format!("{metric_name}-{cardinality}");
        let dataset = VecDataset::new(data_name, data.clone(), metric, true);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = Cakes::new(dataset, Some(seed), criteria);

        let radii = [50, 25, 10, 1];
        println!("Running benchmark for {metric_name} ...");
        for radius in radii {
            let id = BenchmarkId::new("Clustered", radius);
            group.bench_with_input(id, &radius, |b, &radius| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius, rnn::Algorithm::Clustered));
            });
        }

        group.sample_size(10);
        let id = BenchmarkId::new("Linear", radii[0]);
        group.bench_with_input(id, &radii[0], |b, _| {
            b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radii[0], rnn::Algorithm::Linear));
        });

        group.finish();
    }
}

criterion_group!(benches, genomic);
criterion_main!(benches);
