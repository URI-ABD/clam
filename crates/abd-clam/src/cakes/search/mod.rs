//! Entropy-Scaling Search algorithms.

mod knn_breadth_first;
mod knn_depth_first;
mod knn_repeated_rnn;
pub mod rnn_clustered;

use distances::Number;
use rayon::prelude::*;

use crate::new_core::{cluster::Ball, Cluster, Dataset, LinearSearch, ParDataset, ParLinearSearch};

/// A `Cluster` that can be searched.
pub trait Searchable<U: Number>: Cluster<U> + Sized {
    /// Search for the `query` in the `data` using the given `algorithm`.
    ///
    /// # Arguments
    ///
    /// - `data` - The dataset to search.
    /// - `query` - The query to search around.
    /// - `algorithm` - The algorithm to use for the search.
    ///
    /// # Returns
    ///
    /// The `(index, distance)` pairs for the nearest neighbors.
    fn search<I, D: Dataset<I, U>>(&self, data: &D, query: &I, algorithm: Algorithm<U>) -> Vec<(usize, U)> {
        algorithm.search(data, self, query)
    }

    /// Search for several `queries` in the `data` using the given `algorithm`.
    fn batch_search<I, D: Dataset<I, U>>(
        &self,
        data: &D,
        queries: &[I],
        algorithm: Algorithm<U>,
    ) -> Vec<Vec<(usize, U)>> {
        queries
            .iter()
            .map(|query| self.search(data, query, algorithm))
            .collect()
    }
}

/// A `Cluster` that can be searched in parallel.
pub trait ParSearchable<U: Number>: Searchable<U> + Send + Sync {
    /// Parallel version of `Searchable::search`.
    fn par_search<I: Send + Sync, D: ParDataset<I, U>>(
        &self,
        data: &D,
        query: &I,
        algorithm: Algorithm<U>,
    ) -> Vec<(usize, U)> {
        algorithm.par_search(data, self, query)
    }

    /// Parallel version of `Searchable::batch_search`. This only provides
    /// parallelism for the queries, not the search itself.
    fn par_batch_search<I: Send + Sync, D: ParDataset<I, U>>(
        &self,
        data: &D,
        queries: &[I],
        algorithm: Algorithm<U>,
    ) -> Vec<Vec<(usize, U)>> {
        queries
            .par_iter()
            .map(|query| self.search(data, query, algorithm))
            .collect()
    }

    /// Parallel version of `ParSearchable::par_search`. This provides
    /// parallelism for both the queries and the search itself.
    fn par_batch_par_search<I: Send + Sync, D: ParDataset<I, U>>(
        &self,
        data: &D,
        queries: &[I],
        algorithm: Algorithm<U>,
    ) -> Vec<Vec<(usize, U)>>
    where
        U: Send + Sync,
    {
        queries
            .par_iter()
            .map(|query| self.par_search(data, query, algorithm))
            .collect()
    }
}

impl<U: Number> Searchable<U> for Ball<U> {}

impl<U: Number> ParSearchable<U> for Ball<U> {}

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
    /// Search for the `query` in the `data` using the given `root` and `algorithm`.
    fn search<I, D: Dataset<I, U>, C: Cluster<U>>(self, data: &D, root: &C, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::search(data, root, query, radius),
            Self::KnnRepeatedRnn(k, max_multiplier) => knn_repeated_rnn::search(data, root, query, k, max_multiplier),
            Self::KnnBreadthFirst(k) => knn_breadth_first::search(data, root, query, k),
            Self::KnnDepthFirst(k) => knn_depth_first::search(data, root, query, k),
        }
    }

    /// Parallel search for the `query` in the `data` using the given `root` and `algorithm`.
    fn par_search<I: Send + Sync, D: ParDataset<I, U>, C: Cluster<U> + Send + Sync>(
        self,
        data: &D,
        root: &C,
        query: &I,
    ) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => rnn_clustered::par_search(data, root, query, radius),
            Self::KnnRepeatedRnn(k, max_multiplier) => {
                knn_repeated_rnn::par_search(data, root, query, k, max_multiplier)
            }
            Self::KnnBreadthFirst(k) => knn_breadth_first::par_search(data, root, query, k),
            Self::KnnDepthFirst(k) => knn_depth_first::par_search(data, root, query, k),
        }
    }

    /// Linear search for the `query` in the `data`.
    pub fn linear_search<I, D: Dataset<I, U> + LinearSearch<I, U>>(self, data: &D, query: &I) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.rnn(query, radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnBreadthFirst(k) | Self::KnnDepthFirst(k) => data.knn(query, k),
        }
    }

    /// Parallel linear search for the `query` in the `data`.
    pub fn par_linear_search<I: Send + Sync, D: ParDataset<I, U> + ParLinearSearch<I, U>>(
        self,
        data: &D,
        query: &I,
    ) -> Vec<(usize, U)> {
        match self {
            Self::RnnClustered(radius) => data.par_rnn(query, radius),
            Self::KnnRepeatedRnn(k, _) | Self::KnnBreadthFirst(k) | Self::KnnDepthFirst(k) => data.par_knn(query, k),
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
}

#[cfg(test)]
pub mod tests {
    use rand::prelude::*;
    use test_case::test_case;

    use crate::new_core::{cluster::ParPartition, FlatVec, Metric};

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

    #[test_case(1_000, 10; "1k-10")]
    #[test_case(10_000, 10; "10k-10")]
    #[test_case(100_000, 10; "100k-10")]
    #[test_case(1_000, 100; "1k-100")]
    #[test_case(10_000, 100; "10k-100")]
    #[test_case(100_000, 100; "100k-100")]
    fn vectors(car: usize, dim: usize) -> Result<(), String> {
        let seed = 42;

        let mut data = gen_random_data(car, dim, 10.0, seed)?;
        let query = &vec![0.0; dim];

        let criteria = |c: &Ball<f32>| c.cardinality() > 1;
        let seed = Some(seed);
        let (root, _) = Ball::par_new_tree_and_permute(&mut data, &criteria, seed);

        let mut algs: Vec<(Algorithm<f32>, fn(Vec<(usize, f32)>, Vec<(usize, f32)>, &str) -> bool)> = vec![];

        for radius in [0.1, 1.0] {
            algs.push((Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 100] {
            algs.push((Algorithm::KnnRepeatedRnn(k, 2.0), check_search_by_distance));
            algs.push((Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        for (alg, checker) in algs {
            let true_hits = alg.par_linear_search(&data, query);

            if car < 100_000 {
                let pred_hits = root.search(&data, query, alg);
                checker(true_hits.clone(), pred_hits, &alg.name());
            }

            let pred_hits = root.par_search(&data, query, alg);
            checker(true_hits, pred_hits, &alg.name());
        }
        Ok(())
    }

    #[test]
    fn strings() -> Result<(), String> {
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

        let distance_fn = |a: &String, b: &String| distances::strings::levenshtein::<u16>(a, b);
        let metric = Metric::new(distance_fn, true);
        let mut data = FlatVec::new(data, metric)?.with_metadata(metadata)?;

        let criteria = |c: &Ball<u16>| c.cardinality() > 1;
        let seed = Some(42);
        let (root, _) = Ball::par_new_tree_and_permute(&mut data, &criteria, seed);

        let mut algs: Vec<(Algorithm<u16>, fn(Vec<(usize, u16)>, Vec<(usize, u16)>, &str) -> bool)> = vec![];

        let query = &seed_string;
        for radius in [4, 8, 16] {
            algs.push((Algorithm::RnnClustered(radius), check_search_by_index));
        }
        for k in [1, 10, 20] {
            algs.push((Algorithm::KnnRepeatedRnn(k, 2), check_search_by_distance));
            algs.push((Algorithm::KnnBreadthFirst(k), check_search_by_distance));
            algs.push((Algorithm::KnnDepthFirst(k), check_search_by_distance));
        }

        for (alg, checker) in algs {
            let true_hits = alg.par_linear_search(&data, query);

            let pred_hits = root.search(&data, query, alg);
            checker(true_hits.clone(), pred_hits, &alg.name());

            let pred_hits = root.par_search(&data, query, alg);
            checker(true_hits, pred_hits, &alg.name());
        }

        Ok(())
    }
}
