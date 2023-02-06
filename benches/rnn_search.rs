pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;

use utils::anomaly_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("RnnSearch");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds
        .sample_size(100);

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        if data_name != "cover" {
            continue;
        }

        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        let queries = (0..1000)
            .map(|i| dataset.get(i % dataset.cardinality()))
            .collect::<Vec<_>>();

        let metric = clam::metric::Euclidean { is_expensive: false };
        let space = clam::TabularSpace::new(&dataset, &metric);
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let radius = cakes.radius();
        let radii_factors = (50..100).step_by(50);

        println!(
            "Running rnn-search with {} queries on {data_name} under euclidean distance with {} cardinality and {} dimensionality.",
            queries.len(),
            dataset.cardinality(),
            dataset.dimensionality()
        );

        let bench_name = format!("{data_name}-euclidean");
        for factor in radii_factors {
            let queries_radii = queries
                .iter()
                .map(|&q| (q, radius / factor.as_f64()))
                .collect::<Vec<_>>();
            group.bench_function(&bench_name, |b| {
                b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries_radii))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
