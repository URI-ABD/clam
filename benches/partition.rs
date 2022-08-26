pub mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;

use utils::anomaly_readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Partition");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0));
        .sample_size(10);

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        if dataset.cardinality() > 50_000 {
            continue;
        }

        let euclidean = metric_from_name::<f32, f32>("euclidean", false).unwrap();
        // let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);

        for use_cache in [false, true] {
            if use_cache {
                continue;
            }

            let bench_name = format!("{}-distance-cache-{}", data_name, use_cache);
            group.bench_with_input(&bench_name, &use_cache, |b, &use_cache| {
                let space = clam::TabularSpace::new(&dataset, euclidean.as_ref(), use_cache);
                b.iter_with_large_drop(|| Cluster::new_root(&space).build().partition(&partition_criteria, true))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);
