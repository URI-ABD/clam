extern crate clam;

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};

use clam::dataset::{Dataset, RowMajor};
use clam::search::Search;
use clam::utils::{ANN_DATASETS, CHAODA_DATASETS, read_ann_data, read_chaoda_data};

#[allow(dead_code)]
fn apogee_chess(c: &mut Criterion) {
    let (name, metric) = ("apogee", "euclidean");
    let (train, test) = read_ann_data(name).unwrap();
    let train_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::new(train, metric, true).unwrap());
    let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::new(test, metric, true).unwrap());
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
                    search.rnn(test_dataset.instance(q), Some(radius));
                }
            })
        });
    }
}

#[allow(dead_code)]
fn ann_benchmarks(c: &mut Criterion) {
    for (name, metric) in ANN_DATASETS.iter() {
        let (train, test) = read_ann_data(name).unwrap();
        let train_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::<f32, f32>::new(train, metric, true).unwrap());
        let test_dataset: Arc<dyn Dataset<f32, f32>> = Arc::new(RowMajor::<f32, f32>::new(test, metric, true).unwrap());
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
                        search.rnn(test_dataset.instance(q), Some(radius));
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
        let dataset: Arc<dyn Dataset<f64, f64>> = Arc::new(RowMajor::new(data, "euclidean", true).unwrap());
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
                        search.rnn(search.dataset.instance(q), Some(radius));
                    }
                })
            });
        }
        break;
    }
}

// criterion_group!(benches, apogee_chess);
// criterion_group!(benches, ann_benchmarks);
criterion_group!(benches, chess_chaoda);
criterion_main!(benches);
