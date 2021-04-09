extern crate clam;

use std::path::PathBuf;
use std::sync::Arc;

use criterion::{criterion_group, criterion_main, Criterion};

use clam::prelude::*;
use clam::sample_datasets::{Fasta, RowMajor};
use clam::cakes::Search;
use clam::utils::{read_ann_data, read_chaoda_data, ANN_DATASETS, CHAODA_DATASETS};

#[allow(dead_code)]
fn apogee_chess(c: &mut Criterion) {
    let (name, metric) = ("apogee", "euclidean");
    let (train, test) = read_ann_data(name).unwrap();
    let train_dataset: Arc<dyn Dataset<f32, f32>> =
        Arc::new(RowMajor::new(train, metric, true).unwrap());
    let test_dataset: Arc<dyn Dataset<f32, f32>> =
        Arc::new(RowMajor::new(test, metric, true).unwrap());
    let search = Search::build(Arc::clone(&train_dataset), Some(50));

    for &radius in [4000_f32, 2000., 1000.].iter() {
        let message = [
            format!("dataset: {:?}, ", name),
            format!("shape: {:?}, ", train_dataset.shape()),
            format!("radius {:}, ", radius),
            format!("num_queries 100."),
        ]
        .join("");
        println!("{}", message);
        let id = &format!("{}, radius {:}", name, radius)[..];
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
fn ann_benchmarks(c: &mut Criterion) {
    for (name, metric) in ANN_DATASETS[1..2].iter() {
        let (train, test) = read_ann_data(name).unwrap();
        let train_dataset: Arc<dyn Dataset<f32, f32>> =
            Arc::new(RowMajor::<f32, f32>::new(train, metric, true).unwrap());
        let test_dataset: Arc<dyn Dataset<f32, f32>> =
            Arc::new(RowMajor::<f32, f32>::new(test, metric, true).unwrap());
        let search = Search::build(Arc::clone(&train_dataset), Some(50));

        for f in 2..5 {
            let radius = search.diameter() * (10_f32).powi(-f);
            let message = [
                format!("dataset: {:?}, ", name),
                format!("shape: {:?}, ", train_dataset.shape()),
                format!("radius fraction {}, ", -f),
                format!("radius {:.2e}, ", radius),
                format!("num_queries 100."),
            ]
            .join("");
            println!("{}", message);
            let id = &format!("{}, fraction {}", name, -f)[..];
            c.bench_function(id, |b| {
                b.iter(|| {
                    for q in 0..100 {
                        search.rnn(&test_dataset.instance(q), Some(radius));
                    }
                })
            });
        }
    }
}

#[allow(dead_code)]
fn chess_chaoda(c: &mut Criterion) {
    for &name in CHAODA_DATASETS.iter() {
        let (data, _) = read_chaoda_data(name).unwrap();
        let dataset: Arc<dyn Dataset<f64, f64>> =
            Arc::new(RowMajor::new(data, "euclidean", true).unwrap());
        let search = Search::build(Arc::clone(&dataset), Some(50));

        for f in 2..5 {
            let fraction = (10_f64).powi(-f);
            let radius: f64 = search.diameter() * fraction;
            let message = [
                format!("dataset: {:?}, ", name),
                format!("shape: {:?}, ", dataset.shape()),
                format!("radius fraction {}, ", -f),
                format!("radius {:.2e}, ", radius),
                format!("num_queries 100."),
            ]
            .join("");
            println!("{}", message);
            let id = &format!("{}, fraction {}", name, -f)[..];
            c.bench_function(id, |b| {
                b.iter(|| {
                    for q in 0..100 {
                        search.rnn(&search.dataset.instance(q), Some(radius));
                    }
                })
            });
        }
        break;
    }
}

#[allow(dead_code)]
fn silva_ssu_ref_hamming(c: &mut Criterion) {
    let silva_dir: PathBuf = [r"/data", "abd", "ann_data"].iter().collect();

    let mut train_path = silva_dir.clone();
    train_path.push("silva-SSU-Ref-train");
    train_path.set_extension("npy");
    let train_shape = &[2_214_740, 50_000];
    let silva_train: Arc<dyn Dataset<u8, u64>> =
        Arc::new(Fasta::new("hamming", train_path, train_shape, 128).unwrap());

    let mut test_path = silva_dir.clone();
    test_path.push("silva-SSU-Ref-test");
    test_path.set_extension("npy");
    let test_shape = &[10_000, 50_000];
    let silva_test: Arc<dyn Dataset<u8, u64>> =
        Arc::new(Fasta::new("hamming", test_path, test_shape, 128).unwrap());

    let search = Search::build(Arc::clone(&silva_train), Some(100));

    for &radius in [50, 100, 500, 1000, 2000].iter() {
        let message = [
            "silva-SSU-Ref".to_string(),
            format!("shape: {:?}, ", silva_train.shape()),
            format!("radius {}, ", radius),
            format!("num_queries 100."),
        ]
        .join("");
        println!("{}", message);
        let id = &format!("silva-SSU-Ref, radius {}", radius)[..];
        c.bench_function(id, |b| {
            b.iter(|| {
                for i in 0..100 {
                    search.rnn(&silva_test.instance(i), Some(radius));
                }
            })
        });
    }
}

// criterion_group!(benches, apogee_chess);
criterion_group!(benches, ann_benchmarks);
// criterion_group!(benches, chess_chaoda);
// criterion_group!(benches, silva_ssu_ref_hamming);
criterion_main!(benches);
