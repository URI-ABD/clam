// extern crate clam;

// use std::sync::Arc;

// use criterion::{criterion_group, criterion_main, Criterion};

// use clam::dataset::Tabular;
// use clam::prelude::*;
// use clam::utils;
// use clam::Cakes;

// #[allow(dead_code)]
// fn cakes_apogee(c: &mut Criterion) {
//     let mut group = c.benchmark_group("cakes_apogee");
//     group.sample_size(30);

//     let (name, metric) = ("apogee", metric_from_name("euclidean").unwrap());
//     let (train, test) = utils::read_ann_data::<f32, f32>(name).unwrap();
//     let train_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(Tabular::new(Arc::new(train), Arc::clone(&metric), true));
//     let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(Tabular::new(Arc::new(test), metric, true));
//     let search = Cakes::build(Arc::clone(&train_dataset), Some(50), None);
//     let num_queries = 1000;

//     for &radius in [4000., 2000., 1000.].iter() {
//         println!(
//             "{}, ({}, {}), {:.2e}, {:?}",
//             name,
//             train_dataset.cardinality(),
//             train_dataset.dimensionality(),
//             radius,
//             num_queries
//         );
//         group.bench_function(format!("radius_{}", radius), |b| {
//             b.iter(|| {
//                 for q in 0..num_queries {
//                     search.rnn(&test_dataset.get(q), Some(radius));
//                 }
//             })
//         });
//     }
// }

// #[allow(dead_code)]
// fn cakes_ann_benchmarks(c: &mut Criterion) {
//     let mut group = c.benchmark_group("cakes_ann");
//     group.sample_size(30);

//     for (name, metric) in utils::ANN_DATASETS.iter() {
//         if !name.eq(&"fashion-mnist") {
//             continue;
//         }

//         let metric = metric_from_name(metric).unwrap();
//         let (train, test) = utils::read_ann_data::<f32, f32>(name).unwrap();
//         let train_dataset: Arc<dyn Dataset<f32, f32>> =
//             Arc::new(Tabular::<f32, f32>::new(Arc::new(train), Arc::clone(&metric), true));
//         let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(Tabular::<f32, f32>::new(Arc::new(test), metric, true));
//         let search = Cakes::build(Arc::clone(&train_dataset), Some(50), None);

//         for f in 3..6 {
//             let radius = search.diameter() * (10_f32).powi(-f);
//             println!(
//                 "{}, ({}, {}), {:.2e}, {:?}",
//                 name,
//                 train_dataset.cardinality(),
//                 train_dataset.dimensionality(),
//                 radius,
//                 test_dataset.cardinality()
//             );
//             group.bench_function(format!("radius_{}", radius), |b| {
//                 b.iter(|| {
//                     for q in 0..test_dataset.cardinality() {
//                         search.rnn(&test_dataset.get(q), Some(radius));
//                     }
//                 })
//             });
//         }
//     }
// }

// TODO: (Najib)
// #[allow(dead_code)]
// fn cakes_chaoda_datasets(c: &mut Criterion) {
//     for &name in CHAODA_DATASETS.iter() {
//         let mut group = c.benchmark_group(format!("cakes_chaoda_{}", name));
//         group.sample_size(30);

//         if name != "annthyroid" {
//             continue;
//         }

//         let (data, _) = read_chaoda_data(name).unwrap();
//         let metric = metric_from_name("euclidean").unwrap();
//         let dataset: Arc<dyn Dataset<f64, f64>> =
//             Arc::new(RowMajor::new(Arc::new(data), metric, true));
//         let search = Cakes::build(Arc::clone(&dataset), Some(50), None);
//         let num_queries = 1000;

//         for (i, &f) in [0.05, 0.02, 1e-2, 1e-3, 1e-4].iter().enumerate() {
//             let radius: f64 = search.diameter() * f;
//             println!(
//                 "{:?}, ({}, {}), radius: {:.2e}, num_queries: {:}.",
//                 name,
//                 dataset.cardinality(),
//                 dataset.dimensionality(),
//                 radius,
//                 num_queries
//             );
//             group.bench_function(format!("{}", i), |b| {
//                 b.iter(|| {
//                     for q in 0..num_queries {
//                         let q = q % dataset.cardinality();
//                         search.rnn(&search.dataset.instance(q), Some(radius));
//                     }
//                 })
//             });
//         }
//         group.finish();
//     }
// }

// criterion_group!(benches, cakes_apogee);
// criterion_group!(benches, cakes_ann_benchmarks);
// criterion_group!(benches, cakes_chaoda_datasets);
// criterion_main!(benches);

pub fn main() {
    println!("Hello from CLAM benches!")
}
