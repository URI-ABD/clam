use std::collections::HashSet;

use rayon::prelude::*;

use distances::Number;
use symagen::random_data;

use abd_clam::{rnn, Cakes, Dataset};

pub mod anomaly_readers;
pub mod search_readers;

#[allow(clippy::type_complexity)]
#[must_use]
pub fn make_data(n: usize, d: usize, q: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, String) {
    let min_val = 0.;
    let max_val = 1.;
    let data = random_data::random_f32(n * 1_000, d, min_val, max_val, 42);
    let queries = random_data::random_f32(q, d, min_val, max_val, 0);
    let name = format!("{n}k-{d}");

    (data, queries, name)
}

pub fn check_search<T: Send + Sync + Copy, U: Number, D: Dataset<T, U>>(queries: &[T], cakes: &Cakes<T, U, D>, r: U) {
    let iqp = queries
        .par_iter()
        .enumerate()
        .map(|(i, &query)| {
            let naive = cakes
                .rnn_search(query, r, rnn::Algorithm::Linear)
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let rnn = cakes
                .rnn_search(query, r, rnn::Algorithm::Clustered)
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

    let naive = naive.iter().copied().collect::<HashSet<usize>>();
    if naive.len() != cakes.len() {
        return Some(format!(
            "Got duplicate indices in naive: {} vs {}",
            naive.len(),
            cakes.len()
        ));
    }

    let cakes = cakes.iter().copied().collect::<HashSet<usize>>();
    if naive.len() != cakes.len() {
        return Some(format!(
            "Got duplicate indices in cakes: {} vs {}",
            naive.len(),
            cakes.len()
        ));
    }

    let common = naive.intersection(&cakes).count();
    if common == naive.len() {
        None
    } else {
        let recall = common.as_f64() / naive.len().as_f64();
        Some(format!(
            "Got a mismatch in results between naive and cakes: recall = {recall:.12}"
        ))
    }
}
