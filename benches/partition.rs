mod utils;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::Criterion;

use clam::prelude::*;
use utils::anomaly_readers;

fn partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("Partition");
    group.significance_level(0.05).sample_size(30);
    // .measurement_time(std::time::Duration::new(60, 0));

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        let euclidean = metric_from_name::<f32, f32>("euclidean").unwrap();
        let log_cardinality = (dataset.cardinality() as f64).log2() as usize;
        let partition_criteria = clam::criteria::PartitionCriteria::new(true).with_min_cardinality(1 + log_cardinality);

        for use_cache in [false, true] {
            let bench_name = format!("{}-distance-cache-{}", data_name, use_cache);
            group.bench_with_input(&bench_name, &use_cache, |b, &use_cache| {
                let space = clam::TabularSpace::new(&dataset, euclidean, use_cache);
                b.iter_with_large_drop(|| Cluster::new_root(&space).build().partition(&partition_criteria, true))
            });
        }
        if dataset.cardinality() > 0 {
            break;
        }
    }

    group.finish();
}

criterion_group!(benches, partition);
criterion_main!(benches);
