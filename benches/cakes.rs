extern crate clam;

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};

use clam::dataset::RowMajor;
use clam::prelude::*;
use clam::utils::{read_ann_data, read_chaoda_data, ANN_DATASETS, CHAODA_DATASETS};
use clam::Cakes;

#[allow(dead_code)]
fn cakes_apogee(c: &mut Criterion) {
    let (name, metric) = ("apogee", "euclidean");
    let (train, test) = read_ann_data(name).unwrap();
    let train_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::new(train, metric, true).unwrap());
    let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::new(test, metric, true).unwrap());
    let search = Cakes::build(Arc::clone(&train_dataset), Some(50), None);

    for &radius in [4000_f32, 2000., 1000.].iter() {
        let message = [
            format!("dataset: {:?}, ", name),
            format!("shape: {:?}, ", train_dataset.shape()),
            format!("radius {:}, ", radius),
            format!("num_queries 100."),
        ]
        .join("");
        println!("{}", message);
        let id = &format!("apogee_{}_radius_{}", name, radius)[..];
        c.bench_function(id, |b| {
            b.iter(|| {
                for q in 0..100 {
                    search.rnn(&test_dataset.instance(q), Some(radius));
                }
            })
        });
    }
}

#[allow(dead_code)]
fn cakes_ann_benchmarks(c: &mut Criterion) {
    for (name, metric) in ANN_DATASETS.iter() {
        if !name.eq(&"fashion-mnist") {
            continue;
        }

        let (train, test) = read_ann_data(name).unwrap();
        let train_dataset: Arc<dyn Dataset<f32, f32>> =
            Arc::new(RowMajor::<f32, f32>::new(train, metric, true).unwrap());
        let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::<f32, f32>::new(test, metric, true).unwrap());
        let search = Cakes::build(Arc::clone(&train_dataset), Some(50), None);

        for f in 3..6 {
            let radius = search.diameter() * (10_f32).powi(-f);
            let message = [
                format!("dataset: {:?}, ", name),
                format!("shape: {:?}, ", train_dataset.shape()),
                format!("radius fraction {}, ", -f),
                format!("radius {:.2e}, ", radius),
                format!("num_queries {}.", test_dataset.ninstances()),
            ]
            .join("");
            println!("{}", message);
            let id = &format!("ann_{}_exponent_{}", name, f)[..];
            c.bench_function(id, |b| {
                b.iter(|| {
                    for q in 0..test_dataset.ninstances() {
                        search.rnn(&test_dataset.instance(q), Some(radius));
                    }
                })
            });
        }
    }
}

#[allow(dead_code)]
fn cakes_chaoda_datasets(c: &mut Criterion) {
    for &name in CHAODA_DATASETS.iter() {
        let (data, _) = read_chaoda_data(name).unwrap();
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::new(data, "euclidean", true).unwrap());
        let search = Cakes::build(Arc::clone(&dataset), Some(50), None);

        for f in 2..5 {
            let fraction = (10_f64).powi(-f);
            let radius: f64 = search.diameter() * fraction;
            let message = [
                format!("dataset: {:?}, ", name),
                format!("shape: {:?}, ", dataset.shape()),
                format!("exponent {}, ", -f),
                format!("radius {:.2e}, ", radius),
                format!("num_queries 100."),
            ]
            .join("");
            println!("{}", message);
            let id = &format!("chaoda_{}_exponent_{}", name, f)[..];
            c.bench_function(id, |b| {
                b.iter(|| {
                    for q in 0..1000 {
                        let q = q % dataset.ninstances();
                        search.rnn(&search.dataset.instance(q), Some(radius));
                    }
                })
            });
        }
        break;
    }
}

// criterion_group!(benches, cakes_apogee);
// criterion_group!(benches, cakes_ann_benchmarks);
criterion_group!(benches, cakes_chaoda_datasets);
criterion_main!(benches);
