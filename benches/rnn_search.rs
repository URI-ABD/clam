pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;

use utils::anomaly_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("rnn-search");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds
        .sample_size(100);

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        let queries = (0..1000)
            .map(|i| dataset.get(i % dataset.cardinality()))
            .collect::<Vec<_>>();

        let metric_name = "euclidean";
        let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let radius = cakes.radius();
        let radii_factors = (50..100).step_by(50);

        let bench_name = format!(
            "{}-{}-{}-{}",
            data_name,
            dataset.cardinality(),
            dataset.dimensionality(),
            metric_name
        );
        for factor in radii_factors {
            group.bench_with_input(BenchmarkId::new(&bench_name, factor), &factor, |b, &factor| {
                b.iter_with_large_drop(|| {
                    queries
                        .iter()
                        .map(|&query| cakes.rnn_search(query, radius / (factor as f32)))
                        .collect::<Vec<_>>()
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
