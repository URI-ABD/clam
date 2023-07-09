use criterion::*;

use distances::strings::levenshtein;
use symagen::random_data;

use abd_clam::{cluster::PartitionCriteria, dataset::VecVec, needleman_wunch::nw_distance, search::cakes::CAKES};

fn genomic(c: &mut Criterion) {
    let mut group = c.benchmark_group("genomic".to_string());
    group.significance_level(0.025).sample_size(10);

    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Linear);
    group.plot_config(plot_config);

    group.sampling_mode(SamplingMode::Flat);

    let num_queries = 10;
    group.throughput(Throughput::Elements(num_queries as u64));

    let seed = 42;
    let cardinality = 1_000;
    let (min_len, max_len) = (100, 120);
    let alphabet = "ACGT";

    let data = random_data::random_string(cardinality, min_len, max_len, alphabet, seed);
    let queries = random_data::random_string(num_queries, min_len, max_len, alphabet, seed + 1);
    let data = data.iter().map(|v| v.as_str()).collect::<Vec<_>>();
    let queries = queries.iter().map(|v| v.as_str()).collect::<Vec<_>>();

    #[allow(clippy::type_complexity)]
    let metrics: [(&str, fn(&str, &str) -> u16); 2] = [("levenshtein", levenshtein), ("needleman_wunsch", nw_distance)];

    for (metric_name, metric) in metrics {
        let name = format!("{}-{}", metric_name, cardinality);
        let dataset = VecVec::new(data.clone(), metric, name, true);
        let criteria = PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = CAKES::new(dataset, Some(seed)).build(criteria);

        for radius in [10, 25, 50, 60] {
            let id = BenchmarkId::new(metric_name.to_string(), radius);
            group.bench_with_input(id, &radius, |b, &radius| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries, radius));
            });

            let id = BenchmarkId::new(format!("par-{}", metric_name), radius);
            group.bench_with_input(id, &radius, |b, &radius| {
                b.iter_with_large_drop(|| cakes.par_batch_rnn_search(&queries, radius));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, genomic);
criterion_main!(benches);
