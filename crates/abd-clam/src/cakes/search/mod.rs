//! Entropy scaling search algorithms.

mod rnn_clustered;

use distances::Number;

use crate::{cluster::ParCluster, dataset::ParDataset, Cluster, Dataset};

/// The different algorithms that can be used for search.
///
/// - `RnnClustered` - Ranged Nearest Neighbors search using the tree.
/// - `KnnRepeatedRnn` - K-Nearest Neighbors search using repeated `RnnClustered` searches.
/// - `KnnBreadthFirst` - K-Nearest Neighbors search using a breadth-first sieve.
/// - `KnnDepthFirst` - K-Nearest Neighbors search using a depth-first sieve.
///
/// See the `CAKES` paper for more information on these algorithms.
#[derive(Clone, Copy)]
pub enum Algorithm<U: Number> {
    /// Ranged Nearest Neighbors search using the tree.
    ///
    /// # Parameters
    ///
    /// - `U` - The radius to search within.
    RnnClustered(U),
    /// K-Nearest Neighbors search using repeated `RnnClustered` searches.
    ///
    /// # Parameters
    ///
    /// - `usize` - The number of neighbors to search for.
    /// - `U` - The maximum multiplier for the radius when repeating the search.
    KnnRepeatedRnn(usize, U),
    /// K-Nearest Neighbors search using a breadth-first sieve.
    KnnBreadthFirst(usize),
    /// K-Nearest Neighbors search using a depth-first sieve.
    KnnDepthFirst(usize),
}

impl<U: Number> Algorithm<U> {
    /// Perform the search using the algorithm.
    pub fn search<I, C: Cluster<U>, D: Dataset<I, U>>(&self, data: &D, root: &C, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::search(data, root, query, *radius),
            _ => unimplemented!(),
        }
    }

    /// Parallel version of the `search` method.
    pub fn par_search<I: Send + Sync, C: ParCluster<U>, D: ParDataset<I, U>>(
        &self,
        data: &D,
        root: &C,
        query: &I,
    ) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::par_search(data, root, query, *radius),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use rand::prelude::*;

    use crate::{linear_search::LinearSearch, FlatVec, Metric};

    use super::*;

    pub fn gen_line_data(max: i32) -> Result<FlatVec<i32, u32, usize>, String> {
        let data = (-max..=max).collect::<Vec<_>>();
        let distance_fn = |a: &i32, b: &i32| a.abs_diff(*b);
        let metric = Metric::new(distance_fn, false);
        FlatVec::new(data, metric)
    }

    pub fn gen_grid_data(max: i32) -> Result<FlatVec<(f32, f32), f32, usize>, String> {
        let data = (-max..=max)
            .flat_map(|x| (-max..=max).map(move |y| (x.as_f32(), y.as_f32())))
            .collect::<Vec<_>>();
        let distance_fn = |(x1, y1): &(f32, f32), (x2, y2): &(f32, f32)| (x1 - x2).hypot(y1 - y2);
        let metric = Metric::new(distance_fn, false);
        FlatVec::new(data, metric)
    }

    #[allow(dead_code)]
    pub fn gen_random_data(
        car: usize,
        dim: usize,
        max: f32,
        seed: u64,
    ) -> Result<FlatVec<Vec<f32>, f32, usize>, String> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let data = symagen::random_data::random_tabular(car, dim, -max, max, &mut rng);
        let distance_fn = |a: &Vec<f32>, b: &Vec<f32>| distances::vectors::euclidean(a, b);
        let metric = Metric::new(distance_fn, false);
        FlatVec::new(data, metric)
    }

    pub fn check_search_by_index<U: Number>(
        mut true_hits: Vec<(usize, U)>,
        mut pred_hits: Vec<(usize, U)>,
        name: &str,
    ) -> bool {
        true_hits.sort_by_key(|(i, _)| *i);
        pred_hits.sort_by_key(|(i, _)| *i);

        for ((i, p), (j, q)) in true_hits.into_iter().zip(pred_hits) {
            let msg = format!("Failed {name} i: {i}, j: {j}, p: {p}, q: {q}");
            assert_eq!(i, j, "{msg}");
            assert!(p.abs_diff(q) <= U::EPSILON, "{msg}");
        }

        true
    }

    pub fn check_search_by_distance<U: Number>(
        mut true_hits: Vec<(usize, U)>,
        mut pred_hits: Vec<(usize, U)>,
        name: &str,
    ) -> bool {
        true_hits.sort_by(|(_, p), (_, q)| p.partial_cmp(q).unwrap_or(core::cmp::Ordering::Greater));
        pred_hits.sort_by(|(_, p), (_, q)| p.partial_cmp(q).unwrap_or(core::cmp::Ordering::Greater));

        for (i, (&(_, p), &(_, q))) in true_hits.iter().zip(pred_hits.iter()).enumerate() {
            assert!(
                p.abs_diff(q) <= U::EPSILON,
                "Failed {name} i-th: {i}, p: {p}, q: {q} in {true_hits:?} vs {pred_hits:?}"
            );
        }

        true
    }

    pub fn check_rnn<I: Send + Sync, U: Number, C: ParCluster<U>>(
        root: &C,
        data: &FlatVec<I, U, usize>,
        query: &I,
        radius: U,
    ) -> bool {
        let true_hits = data.rnn(query, radius);

        let pred_hits = rnn_clustered::search(data, root, query, radius);
        assert_eq!(pred_hits.len(), true_hits.len(), "Rnn search failed: {pred_hits:?}");
        check_search_by_index(true_hits.clone(), pred_hits, "RnnClustered");

        let pred_hits = rnn_clustered::par_search(data, root, query, radius);
        assert_eq!(
            pred_hits.len(),
            true_hits.len(),
            "Parallel Rnn search failed: {pred_hits:?}"
        );
        check_search_by_index(true_hits, pred_hits, "Par RnnClustered");

        true
    }

    #[allow(dead_code)]
    pub fn check_knn<I: Send + Sync, U: Number, C: ParCluster<U>>(
        root: &C,
        data: &FlatVec<I, U, usize>,
        query: &I,
        k: usize,
        radius: U,
    ) -> bool {
        let true_hits = data.knn(query, k);

        let pred_hits = rnn_clustered::search(data, root, query, radius);
        assert_eq!(pred_hits.len(), true_hits.len(), "Knn search failed: {pred_hits:?}");
        check_search_by_distance(true_hits.clone(), pred_hits, "KnnClustered");

        let pred_hits = rnn_clustered::par_search(data, root, query, radius);
        assert_eq!(
            pred_hits.len(),
            true_hits.len(),
            "Parallel Knn search failed: {pred_hits:?}"
        );
        check_search_by_distance(true_hits, pred_hits, "Par KnnClustered");

        true
    }
}
