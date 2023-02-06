use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;
use float_cmp::approx_eq;

use clam::prelude::*;

pub mod utils;
use utils::anomaly_readers;

fn cakes(c: &mut Criterion) {
    // ChaCha8Rng::seed_from_u64(42);
    // println!("{:?}", )

    let mut group = c.benchmark_group("knn-search");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds
        .sample_size(10);

    for &data_name in anomaly_readers::ANOMALY_DATASETS.iter() {
        // if data_name != "http" {
        //     continue;
        // }

        let (features, _) = anomaly_readers::read_anomaly_data(data_name, true).unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        let queries = (0..100)
            .map(|i| dataset.get(i % dataset.cardinality()))
            .collect::<Vec<_>>();

        let metric_name = "euclidean";
        let metric = metric_from_name::<f32>(metric_name, false).unwrap();
        let space = clam::TabularSpace::new(&dataset, metric.as_ref());
        let partition_criteria = clam::PartitionCriteria::new(true).with_min_cardinality(1);
        let cakes = clam::CAKES::new(&space).build(&partition_criteria);

        let ks = [1, 10, 100];

        let bench_name = format!(
            "{}-{}-{}-{}",
            data_name,
            dataset.cardinality(),
            dataset.dimensionality(),
            metric_name
        );
        for k in ks {
            if k > dataset.cardinality() {
                continue;
            }

            queries.iter().for_each(|&query| {
                // let indices = cakes.knn_search(query, k);
                let indices = cakes
                    .knn_by_rnn(query, k)
                    .into_iter()
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();
                assert!(indices.len() >= k, "{} vs {}", indices.len(), k);
                if indices.len() > k {
                    let mut distances = space.query_to_many(query, &indices);
                    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let kth = distances[k - 1];
                    assert!(
                        distances[k..].iter().all(|d| approx_eq!(f64, *d, kth)),
                        "{:?}",
                        &distances
                    );
                }
            });

            group.bench_with_input(BenchmarkId::new(&bench_name, k), &k, |b, &k| {
                b.iter_with_large_drop(|| {
                    queries
                        .iter()
                        // .map(|&query| cakes.knn_search(query, k))
                        .map(|&query| cakes.knn_by_rnn(query, k))
                        .collect::<Vec<_>>()
                })
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
