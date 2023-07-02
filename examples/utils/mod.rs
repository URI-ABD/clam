use std::collections::HashSet;

use rayon::prelude::*;

use abd_clam::dataset::Dataset;
use abd_clam::number::Number;
use abd_clam::search::cakes::CAKES;
use abd_clam::utils::synthetic_data;

pub mod anomaly_readers;
pub mod distances;
pub mod search_readers;

#[allow(clippy::type_complexity)]
pub fn make_data(n: usize, d: usize, q: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, String) {
    let min_val = 0.;
    let max_val = 1.;
    let data = synthetic_data::random_f32(n * 1_000, d, min_val, max_val, 42);
    let queries = synthetic_data::random_f32(q, d, min_val, max_val, 0);
    let name = format!("{n}k-{d}");

    (data, queries, name)
}

pub fn check_search<T: Number, U: Number, D: Dataset<T, U>>(queries: &[&Vec<T>], cakes: &CAKES<T, U, D>, r: U) {
    let iqp = queries
        .par_iter()
        .enumerate()
        .map(|(i, &query)| {
            let naive = cakes
                .linear_search(query, r, None)
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let rnn = cakes
                .rnn_search(query, r)
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            (i, query, check_exactness(&naive, &rnn))
        })
        .find_first(|(_, _, p)| p.is_some());
    if let Some((i, _, problem)) = iqp {
        let problem = problem.unwrap();
        println!("Problem at index {i}: {problem}");
    }
}

fn check_exactness(naive: &[usize], cakes: &[usize]) -> Option<String> {
    if naive.len() != cakes.len() {
        return Some(format!(
            "Different number of results: {} vs {}",
            naive.len(),
            cakes.len()
        ));
    }

    let naive = HashSet::<usize>::from_iter(naive.iter().copied());
    if naive.len() != cakes.len() {
        return Some(format!(
            "Got duplicate indices in naive: {} vs {}",
            naive.len(),
            cakes.len()
        ));
    }

    let cakes = HashSet::<usize>::from_iter(cakes.iter().copied());
    if naive.len() != cakes.len() {
        return Some(format!(
            "Got duplicate indices in cakes: {} vs {}",
            naive.len(),
            cakes.len()
        ));
    }

    let common = naive.intersection(&cakes).count();
    if common != naive.len() {
        let recall = common.as_f64() / naive.len().as_f64();
        Some(format!(
            "Got a mismatch in results between naive and cakes: recall = {recall:.12}"
        ))
    } else {
        None
    }
}
