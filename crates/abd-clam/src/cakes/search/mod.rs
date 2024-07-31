//! Entropy scaling search algorithms.

mod knn_breadth_first;
mod knn_depth_first;
mod knn_repeated_rnn;
mod rnn_clustered;

use distances::Number;

use crate::{
    cluster::ParCluster,
    dataset::ParDataset,
    linear_search::{LinearSearch, ParLinearSearch},
    Cluster, Dataset,
};

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
            Self::KnnRepeatedRnn(k, max_multiplier) => knn_repeated_rnn::search(data, root, query, *k, *max_multiplier),
            Self::KnnBreadthFirst(k) => knn_breadth_first::search(data, root, query, *k),
            Self::KnnDepthFirst(k) => knn_depth_first::search(data, root, query, *k),
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
            Self::KnnRepeatedRnn(k, max_multiplier) => {
                knn_repeated_rnn::par_search(data, root, query, *k, *max_multiplier)
            }
            Self::KnnBreadthFirst(k) => knn_breadth_first::par_search(data, root, query, *k),
            Self::KnnDepthFirst(k) => knn_depth_first::par_search(data, root, query, *k),
        }
    }

    /// Search via a linear scan.
    pub fn linear_search<I, D: LinearSearch<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.rnn(query, *radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnDepthFirst(k) | Self::KnnBreadthFirst(k) => data.knn(query, *k),
        }
    }

    /// Parallel version of the `linear_search` method
    pub fn par_linear_search<I: Send + Sync, D: ParLinearSearch<I, U>>(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.par_rnn(query, *radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnDepthFirst(k) | Self::KnnBreadthFirst(k) => data.par_knn(query, *k),
        }
    }

    /// Get the name of the algorithm.
    pub fn name(&self) -> String {
        match self {
            Self::RnnClustered(r) => format!("RnnClustered({r})"),
            Self::KnnRepeatedRnn(k, m) => format!("KnnRepeatedRnn({k}, {m})"),
            Self::KnnBreadthFirst(k) => format!("KnnBreadthFirst({k})"),
            Self::KnnDepthFirst(k) => format!("KnnDepthFirst({k})"),
        }
    }

    /// Get the name of the variant of algorithm.
    pub const fn variant_name(&self) -> &str {
        match self {
            Self::RnnClustered(_) => "RnnClustered",
            Self::KnnRepeatedRnn(_, _) => "KnnRepeatedRnn",
            Self::KnnBreadthFirst(_) => "KnnBreadthFirst",
            Self::KnnDepthFirst(_) => "KnnDepthFirst",
        }
    }

    /// Same variant of the algorithm with different parameters.
    #[must_use]
    pub const fn with_params(&self, radius: U, k: usize) -> Self {
        match self {
            Self::RnnClustered(_) => Self::RnnClustered(radius),
            Self::KnnRepeatedRnn(_, m) => Self::KnnRepeatedRnn(k, *m),
            Self::KnnBreadthFirst(_) => Self::KnnBreadthFirst(k),
            Self::KnnDepthFirst(_) => Self::KnnDepthFirst(k),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use distances::Number;
    use rand::prelude::*;
    use test_case::test_case;

    use crate::{
        cakes::{cluster::SquishyBall, OffsetBall},
        Ball, Cluster, FlatVec, Metric, Partition,
    };

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

    pub fn check_search_by_index<U: Number>(
        mut true_hits: Vec<(usize, U)>,
        mut pred_hits: Vec<(usize, U)>,
        name: &str,
        squishy: bool,
    ) -> bool {
        true_hits.sort_by_key(|(i, _)| *i);
        pred_hits.sort_by_key(|(i, _)| *i);
        let squishy = if squishy { "squishy " } else { "" };

        assert_eq!(
            true_hits.len(),
            pred_hits.len(),
            "{squishy}{name}: {true_hits:?} vs {pred_hits:?}"
        );

        for ((i, p), (j, q)) in true_hits.into_iter().zip(pred_hits) {
            let msg = format!("Failed {squishy}{name} i: {i}, j: {j}, p: {p}, q: {q}");
            assert_eq!(i, j, "{msg}");
            assert!(p.abs_diff(q) <= U::EPSILON, "{msg}");
        }

        true
    }

    pub fn check_search_by_distance<U: Number>(
        mut true_hits: Vec<(usize, U)>,
        mut pred_hits: Vec<(usize, U)>,
        name: &str,
        squishy: bool,
    ) -> bool {
        true_hits.sort_by(|(_, p), (_, q)| p.partial_cmp(q).unwrap_or(core::cmp::Ordering::Greater));
        pred_hits.sort_by(|(_, p), (_, q)| p.partial_cmp(q).unwrap_or(core::cmp::Ordering::Greater));
        let squishy = if squishy { "squishy " } else { "" };

        assert_eq!(
            true_hits.len(),
            pred_hits.len(),
            "{squishy}{name}: {true_hits:?} vs {pred_hits:?}"
        );

        for (i, (&(_, p), &(_, q))) in true_hits.iter().zip(pred_hits.iter()).enumerate() {
            assert!(
                p.abs_diff(q) <= U::EPSILON,
                "Failed {squishy}{name} i-th: {i}, p: {p}, q: {q} in {true_hits:?} vs {pred_hits:?}"
            );
        }

        true
    }

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

    #[test_case(1_000, 10; "1k-10")]
    #[test_case(10_000, 10; "10k-10")]
    #[test_case(100_000, 10; "100k-10")]
    #[test_case(1_000, 100; "1k-100")]
    #[test_case(10_000, 100; "10k-100")]
    fn vectors(car: usize, dim: usize) -> Result<(), String> {
        let mut algs: Vec<(
            super::Algorithm<f32>,
            fn(Vec<(usize, f32)>, Vec<(usize, f32)>, &str, bool) -> bool,
        )> = vec![];
        for radius in [0.1, 1.0] {
            algs.push((super::Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 100] {
            algs.push((super::Algorithm::KnnRepeatedRnn(k, 2.0), check_search_by_distance));
            algs.push((super::Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((super::Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        let seed = 42;
        let data = gen_random_data(car, dim, 10.0, seed)?;
        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(seed);
        let query = &vec![0.0; dim];

        let root = Ball::new_tree(&data, &criteria, seed);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for &(alg, checker) in &algs {
            let true_hits = alg.par_linear_search(&data, query);

            if car < 100_000 {
                let pred_hits = alg.search(&data, &root, query);
                checker(true_hits.clone(), pred_hits, &alg.name(), false);
                let pred_hits = alg.search(&data, &squishy_root, query);
                checker(true_hits.clone(), pred_hits, &alg.name(), true);
            }

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);
            let pred_hits = alg.par_search(&data, &squishy_root, query);
            checker(true_hits, pred_hits, &alg.name(), true);
        }

        let mut data = data;
        let root = OffsetBall::from_ball_tree(root, &mut data);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for (alg, checker) in algs {
            let true_hits = alg.par_linear_search(&data, query);

            if car < 100_000 {
                let pred_hits = alg.search(&data, &root, query);
                checker(true_hits.clone(), pred_hits, &alg.name(), false);
                let pred_hits = alg.search(&data, &squishy_root, query);
                checker(true_hits.clone(), pred_hits, &alg.name(), true);
            }

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);
            let pred_hits = alg.par_search(&data, &squishy_root, query);
            checker(true_hits, pred_hits, &alg.name(), true);
        }

        Ok(())
    }

    #[test]
    fn strings() -> Result<(), String> {
        let mut algs: Vec<(
            super::Algorithm<u16>,
            fn(Vec<(usize, u16)>, Vec<(usize, u16)>, &str, bool) -> bool,
        )> = vec![];
        for radius in [4, 8, 16] {
            algs.push((super::Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 20] {
            algs.push((super::Algorithm::KnnRepeatedRnn(k, 2), check_search_by_distance));
            algs.push((super::Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((super::Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        let seed_length = 100;
        let alphabet = "ACTGN".chars().collect::<Vec<_>>();
        let seed_string = symagen::random_edits::generate_random_string(seed_length, &alphabet);
        let penalties = distances::strings::Penalties::default();
        let num_clumps = 16;
        let clump_size = 16;
        let clump_radius = 3_u16;
        let (metadata, data) = symagen::random_edits::generate_clumped_data(
            &seed_string,
            penalties,
            &alphabet,
            num_clumps,
            clump_size,
            clump_radius,
        )
        .into_iter()
        .unzip::<_, _, Vec<_>, Vec<_>>();
        let query = &seed_string;

        let distance_fn = |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b);
        let metric = Metric::new(distance_fn, true);
        let data = FlatVec::new(data, metric)?.with_metadata(metadata)?;

        let criteria = |c: &Ball<u16>| c.cardinality() > 1;
        let seed = Some(42);

        let root = Ball::new_tree(&data, &criteria, seed);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for &(alg, checker) in &algs {
            let true_hits = alg.par_linear_search(&data, query);

            let pred_hits = alg.search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);

            let pred_hits = alg.search(&data, &squishy_root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), true);

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);

            let pred_hits = alg.par_search(&data, &squishy_root, query);
            checker(true_hits, pred_hits, &alg.name(), true);
        }

        let mut data = data;
        let root = OffsetBall::from_ball_tree(root, &mut data);
        let squishy_root = SquishyBall::from_root(root.clone(), &data, true);

        for (alg, checker) in algs {
            let true_hits = alg.par_linear_search(&data, query);

            let pred_hits = alg.search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);

            let pred_hits = alg.search(&data, &squishy_root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), true);

            let pred_hits = alg.par_search(&data, &root, query);
            checker(true_hits.clone(), pred_hits, &alg.name(), false);

            let pred_hits = alg.par_search(&data, &squishy_root, query);
            checker(true_hits, pred_hits, &alg.name(), true);
        }

        Ok(())
    }
}
