mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;
use utils::anomaly_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("CAKES");
    group
        .significance_level(0.05)
        .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds

    let num_queries = 1000;

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();
        let dataset = clam::Tabular::new(&features, data_name.to_string());

        let euclidean = metric_from_name::<f32, f32>("euclidean").unwrap();
        let space = clam::TabularSpace::new(&dataset, euclidean, false);
        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = clam::criteria::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let radius = cakes.diameter();
        let queries: Vec<_> = (0..num_queries)
            .map(|i| dataset.get(i % dataset.cardinality()))
            .collect();

        let radii_factors: Vec<usize> = (1..2)
            .chain((10..41).step_by(10))
            .chain((50..201).step_by(50))
            .chain((250..1001).step_by(250))
            .collect();

        for &factor in radii_factors.iter() {
            group.bench_with_input(BenchmarkId::new(data_name, factor), &factor, |b, &factor| {
                let queries_radii: Vec<_> = queries.iter().map(|q| (q.clone(), radius / (factor as f32))).collect();
                if factor == 1 {
                    b.iter_with_large_drop(|| cakes.batch_linear_search(&queries_radii))
                } else {
                    b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries_radii))
                }
            });
        }
        if dataset.cardinality() > 0 {
            break;
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
