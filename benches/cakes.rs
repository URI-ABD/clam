pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;
use utils::search_readers;

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("CAKES");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds
        .sample_size(10);

    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS.iter() {
        if data_name != "fashion-mnist" {
            continue;
        }

        let (data, queries) = search_readers::read_search_data(data_name).unwrap();
        let dataset = clam::Tabular::new(&data, data_name.to_string());

        let metric = metric_from_name::<f32, f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&dataset, metric.as_ref(), false);
        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let radius = cakes.diameter();
        let radii_factors: Vec<usize> = (1..2)
            .chain((25..100).step_by(25))
            .chain((100..=500).step_by(100))
            .collect();

        let bench_name = format!(
            "{}-{}-{}-{}",
            data_name,
            dataset.cardinality(),
            dataset.dimensionality(),
            metric_name
        );
        for &factor in radii_factors.iter() {
            group.bench_with_input(BenchmarkId::new(&bench_name, factor), &factor, |b, &factor| {
                let queries_radii: Vec<_> = queries.iter().map(|q| (q.clone(), radius / (factor as f32))).collect();
                if factor == 1 {
                    b.iter_with_large_drop(|| cakes.batch_linear_search(&queries_radii))
                } else {
                    b.iter_with_large_drop(|| cakes.batch_rnn_search(&queries_radii))
                }
            });
            if factor > 5 {
                break;
            }
        }
        if dataset.cardinality() > 0 {
            break;
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
