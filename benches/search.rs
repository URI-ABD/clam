extern crate clam;

use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};

use clam::dataset::Dataset;
use clam::search::Search;
use clam::utils::{read_apogee, DATASETS, read_data};

#[allow(dead_code)]
fn apogee_chess(c: &mut Criterion) {
    let name = "apogee";
    let data = read_apogee();
    let dataset = Arc::new(Dataset::new(data, "euclidean", true).unwrap());
    let search = Search::build(Arc::clone(&dataset), Some(50));

    for &f in [2., 1., 0.1, 0.01].iter() {
        let fraction = f / 100.;
        let radius = search.diameter() * fraction;
        let message = [
            format!("dataset: {:?}, ", name),
            format!("shape: {:?}, ", dataset.shape()),
            format!("radius fraction {:.0e}, ", fraction),
            format!("radius {:.2e}, ", radius),
            format!("num_queries 100."),
        ].join("");
        println!("{}", message);
        let id = &format!("{}, fraction {:.0e}", name, fraction)[..];
        c.bench_function(
            id,
            |b| b.iter(|| {
                for q in 0..100 {
                    search.rnn(search.dataset.row(q), Some(radius));
                }
            }),
        );
    }
}

#[allow(dead_code)]
fn chess_chaoda(c: &mut Criterion) {
    for &name in DATASETS.iter() {
        let (data, _) = read_data(name).unwrap();
        let dataset = Arc::new(Dataset::new(data, "euclidean", true).unwrap());
        let search = Search::build(Arc::clone(&dataset), Some(50));

        for f in 2..5 {
            let fraction = (10_f64).powi(-f);
            let radius = search.diameter() * fraction;
            let message = [
                format!("dataset: {:?}, ", name),
                format!("shape: {:?}, ", dataset.shape()),
                format!("radius fraction {:.0e}, ", fraction),
                format!("radius {:.2e}, ", radius),
                format!("num_queries 100."),
            ].join("");
            println!("{}", message);
            let id = &format!("{}, fraction {:.0e}", name, fraction)[..];
            c.bench_function(
                id,
                |b| b.iter(|| {
                    for q in 0..100 {
                        search.rnn(search.dataset.row(q), Some(radius));
                    }
                }),
            );
        }
    }
}

// criterion_group!(benches, apogee_chess);
criterion_group!(benches, chess_chaoda);
criterion_main!(benches);
