use std::collections::HashSet;

use criterion::criterion_group;
use criterion::criterion_main;
use criterion::BenchmarkId;
use criterion::Criterion;

use clam::prelude::*;

pub mod utils;
use utils::search_readers;

#[allow(dead_code)]
fn check_recall(knn: &[usize], rnn: &[usize]) -> bool {
    (knn.len() == rnn.len()) && {
        let knn: HashSet<usize> = HashSet::from_iter(knn.iter().copied());
        let rnn = HashSet::from_iter(rnn.iter().copied());
        knn.intersection(&rnn).count() == knn.len()
    }
}

fn cakes(c: &mut Criterion) {
    let mut group = c.benchmark_group("knn-vs-rnn");
    group
        .significance_level(0.05)
        // .measurement_time(std::time::Duration::new(60, 0)); // 60 seconds
        .sample_size(10);

    for &(data_name, metric_name) in search_readers::SEARCH_DATASETS {
        if data_name == "fashion-mnist" {
            continue;
        }

        let data = search_readers::read_search_data(data_name);
        if data.is_err() {
            log::info!("Could not read {} data. Moving on ...", data_name);
            continue;
        }
        let (features, queries) = data.unwrap();

        let dataset = clam::Tabular::new(&features, data_name.to_string());
        if dataset.cardinality() > 100_000 {
            continue;
        }

        let queries = clam::Tabular::new(&queries, format!("{data_name}-queries"));
        let queries = (0..100)
            .map(|i| queries.get(i % dataset.cardinality()))
            .collect::<Vec<_>>();

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

            // let knn_hits = cakes.batch_knn_search(&queries, k);

            // let radii = queries
            //     .iter()
            //     .zip(knn_hits.iter())
            //     .map(|(&query, indices)| helpers::arg_max(&space.query_to_many(query, indices)).1)
            //     .collect::<Vec<_>>();

            // let rnn_hits = queries
            //     .iter()
            //     .zip(radii.iter())
            //     .map(|(&query, &radius)| {
            //         cakes
            //             .rnn_search(query, radius)
            //             .into_iter()
            //             .map(|(i, _)| i)
            //             .collect::<Vec<_>>()
            //     })
            //     .collect::<Vec<_>>();

            // knn_hits
            //     .iter()
            //     .zip(rnn_hits.iter())
            //     .for_each(|(knn, rnn)| assert!(check_recall(knn, rnn)));

            // let knn_name = format!("{}-knn", &bench_name);
            // group.bench_with_input(BenchmarkId::new(&knn_name, k), &k, |b, &k| {
            //     b.iter_with_large_drop(|| cakes.batch_knn_search(&queries, k))
            // });

            let rnn_name = format!("{}-rnn", &bench_name);
            group.bench_with_input(BenchmarkId::new(&rnn_name, k), &k, |b, _| {
                b.iter_with_large_drop(|| cakes.batch_knn_by_rnn(&queries, k))
            });
        }
    }

    group.finish();
}

criterion_group!(benches, cakes);
criterion_main!(benches);
