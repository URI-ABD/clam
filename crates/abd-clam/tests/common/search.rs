//! Common functions for testing search algorithms.

use abd_clam::{
    cakes::{self, ParSearchAlgorithm, ParSearchable, SearchAlgorithm},
    cluster::ParCluster,
    metric::ParMetric,
};
use distances::Number;

pub fn check_search_by_index<T: Number>(mut true_hits: Vec<(usize, T)>, mut pred_hits: Vec<(usize, T)>, name: &str) {
    true_hits.sort_by_key(|(i, _)| *i);
    pred_hits.sort_by_key(|(i, _)| *i);

    let rest = format!("\n{true_hits:?}\nvs\n{pred_hits:?}");
    assert_eq!(true_hits.len(), pred_hits.len(), "{name}: {rest}");

    for ((i, p), (j, q)) in true_hits.into_iter().zip(pred_hits) {
        let msg = format!("Failed {name} i: {i}, j: {j}, p: {p}, q: {q}");
        assert_eq!(i, j, "{msg} {rest}");
        assert!(p.abs_diff(q) <= T::EPSILON, "{msg} in {rest}.");
    }
}

pub fn check_search_by_distance<T: Number>(mut true_hits: Vec<(usize, T)>, mut pred_hits: Vec<(usize, T)>, name: &str) {
    true_hits.sort_by(|(_, p), (_, q)| p.total_cmp(q));
    pred_hits.sort_by(|(_, p), (_, q)| p.total_cmp(q));

    assert_eq!(
        true_hits.len(),
        pred_hits.len(),
        "{name}: {true_hits:?} vs {pred_hits:?}"
    );

    for (i, (&(_, p), &(_, q))) in true_hits.iter().zip(pred_hits.iter()).enumerate() {
        assert!(
            p.abs_diff(q) <= T::EPSILON,
            "Failed {name} i-th: {i}, p: {p}, q: {q} in {true_hits:?} vs {pred_hits:?}."
        );
    }
}

pub fn check_rnn<I, T, C, M, D, A>(root: &C, data: &D, metric: &M, query: &I, radius: T, alg: &A)
where
    I: core::fmt::Debug + Send + Sync,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParSearchable<I, T, C, M>,
    A: ParSearchAlgorithm<I, T, C, M, D>,
{
    let c_name = std::any::type_name::<C>();

    let true_hits = cakes::RnnLinear(radius).search(data, metric, root, query);

    let pred_hits = alg.search(data, metric, root, query);
    assert_eq!(
        pred_hits.len(),
        true_hits.len(),
        "{} search on {c_name} failed: {pred_hits:?}",
        alg.name()
    );
    check_search_by_index(true_hits.clone(), pred_hits, alg.name());

    let pred_hits = alg.par_search(data, metric, root, query);
    let par_name = format!("Par{}", alg.name());
    assert_eq!(
        pred_hits.len(),
        true_hits.len(),
        "{par_name} search on {c_name} failed: {pred_hits:?}"
    );
    check_search_by_index(true_hits, pred_hits, &par_name);
}

/// Check a k-NN search algorithm.
pub fn check_knn<I, T, D, M, C, A>(root: &C, data: &D, metric: &M, query: &I, k: usize, alg: &A)
where
    I: core::fmt::Debug + Send + Sync,
    T: Number,
    C: ParCluster<T>,
    M: ParMetric<I, T>,
    D: ParSearchable<I, T, C, M>,
    A: ParSearchAlgorithm<I, T, C, M, D>,
{
    let c_name = std::any::type_name::<C>();
    let true_hits = cakes::KnnLinear(k).search(data, metric, root, query);

    let pred_hits = alg.search(data, metric, root, query);
    assert_eq!(
        pred_hits.len(),
        true_hits.len(),
        "{} search on {c_name} failed: pred {pred_hits:?} vs true {true_hits:?}",
        alg.name()
    );
    check_search_by_distance(true_hits.clone(), pred_hits, alg.name());

    let pred_hits = alg.par_search(data, metric, root, query);
    let par_name = format!("Par{}", alg.name());
    assert_eq!(
        pred_hits.len(),
        true_hits.len(),
        "{par_name} search on {c_name} failed: pred {pred_hits:?} vs true {true_hits:?}"
    );
    check_search_by_distance(true_hits, pred_hits, &par_name);
}
