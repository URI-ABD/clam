//! Benchmarks for the suite of ANN-Benchmarks datasets.

use std::{
    collections::HashMap,
    ffi::OsStr,
    path::{Path, PathBuf},
};

use abd_clam::{
    cakes::{HintedDataset, PermutedBall},
    cluster::{adapter::ParBallAdapter, BalancedBall, ParPartition},
    dataset::AssociatesMetadataMut,
    metric::{self, ParMetric},
    Ball, Cluster, Dataset, FlatVec,
};
use criterion::*;
use utils::Row;

mod utils;

/// The datasets available in the `ann-benchmarks` repository.
#[allow(dead_code)]
enum AnnDataset {
    DeepImage,
    FashionMnist,
    Gist,
    Glove25,
    Glove50,
    Glove100,
    Glove200,
    Kosarak,
    Lastfm,
    Mnist,
    Nytimes,
    Sift,
}

impl AnnDataset {
    /// Returns the value of the `AnnDataset` enum from the given name.
    #[allow(dead_code)]
    fn from_name(name: &str) -> Self {
        match name {
            "deep-image" => Self::DeepImage,
            "fashion-mnist" => Self::FashionMnist,
            "gist" => Self::Gist,
            "glove-25" => Self::Glove25,
            "glove-50" => Self::Glove50,
            "glove-100" => Self::Glove100,
            "glove-200" => Self::Glove200,
            "kosarak" => Self::Kosarak,
            "lastfm" => Self::Lastfm,
            "mnist" => Self::Mnist,
            "nytimes" => Self::Nytimes,
            "sift" => Self::Sift,
            _ => panic!("Unknown dataset: {}", name),
        }
    }

    fn to_name(&self) -> &str {
        match self {
            Self::DeepImage => "deep-image",
            Self::FashionMnist => "fashion-mnist",
            Self::Gist => "gist",
            Self::Glove25 => "glove-25",
            Self::Glove50 => "glove-50",
            Self::Glove100 => "glove-100",
            Self::Glove200 => "glove-200",
            Self::Kosarak => "kosarak",
            Self::Lastfm => "lastfm",
            Self::Mnist => "mnist",
            Self::Nytimes => "nytimes",
            Self::Sift => "sift",
        }
    }

    /// Returns the path to the train and test files of the dataset.
    fn paths<P: AsRef<Path>, S: AsRef<OsStr>>(&self, root: &P, ext: S) -> [PathBuf; 2] {
        let name = self.to_name();
        [
            root.as_ref().join(format!("{name}-train")).with_extension(ext.as_ref()),
            root.as_ref().join(format!("{name}-test")).with_extension(ext),
        ]
    }

    /// Reads the dataset from the given root path and extension.
    ///
    /// # Arguments
    ///
    /// - `root`: The root directory of the datasets.
    /// - `ext`: The extension of the files to read.
    /// - `metric`: The metric to use for the training set.
    ///
    /// # Returns
    ///
    /// - The training data on which to build the trees.
    /// - The query data to search for the nearest neighbors.
    fn read<P: AsRef<Path>, S: AsRef<OsStr>>(&self, root: &P, ext: S) -> (FlatVec<Row<f32>, usize>, Vec<Row<f32>>) {
        let [train, test] = self.paths(root, ext);
        println!("Reading train data from {train:?}, {test:?}");
        let test = FlatVec::read_npy(&test)
            .unwrap()
            .transform_items(Row::from)
            .items()
            .to_vec();
        println!("Finished reading test data with {} items.", test.len());
        let train = FlatVec::read_npy(&train)
            .unwrap()
            .with_name(self.to_name())
            .transform_items(Row::from);
        println!("Finished reading train data with {} items.", train.cardinality());
        (train, test)
    }
}

fn run_search<M: ParMetric<Row<f32>, f32>>(
    c: &mut Criterion,
    data: FlatVec<Row<f32>, usize>,
    metric: &M,
    queries: &[Row<f32>],
    radii_fractions: &[f32],
    ks: &[usize],
    seed: Option<u64>,
    multiplier_error: Option<(usize, f32)>,
) {
    let data = if let Some((multiplier, error)) = multiplier_error {
        println!("Augmenting data to {multiplier}x with an error rate of {error:.2}.");

        let mut rows = data.items().iter().map(|r| Row::<f32>::to_vec(r)).collect::<Vec<_>>();
        let new_rows = symagen::augmentation::augment_data(&rows, multiplier, error);
        rows.extend(new_rows);
        let rows = rows.into_iter().map(Row::from).collect();

        let name = format!("augmented-{}-{multiplier}", data.name());
        FlatVec::new(rows)
            .unwrap_or_else(|e| unreachable!("{e}"))
            .with_name(&name)
    } else {
        data
    };

    println!("Creating ball ...");
    let criteria = |c: &Ball<_>| c.cardinality() > 1;
    let ball = Ball::par_new_tree(&data, metric, &criteria, seed);
    let radii = radii_fractions.iter().map(|&r| r * ball.radius()).collect::<Vec<_>>();

    println!("Creating balanced ball ...");
    let criteria = |c: &BalancedBall<_>| c.cardinality() > 1;
    let balanced_ball = BalancedBall::par_new_tree(&data, metric, &criteria, seed).into_ball();

    println!("Adding hints to data ...");
    // let (_, max_radius) = abd_clam::utils::arg_max(&radii).unwrap();
    // let (_, max_k) = abd_clam::utils::arg_max(ks).unwrap();
    let data = data
        .transform_metadata(|&i| (i, HashMap::new()))
        .with_hints_from_tree(&ball, metric)
        .with_hints_from_tree(&balanced_ball, metric);
    // .with_hints_from(metric, &balanced_ball, max_radius, max_k);

    println!("Creating permuted ball ...");
    let (perm_ball, perm_data) = PermutedBall::par_from_ball_tree(ball.clone(), data.clone(), metric);
    println!("Creating permuted balanced ball ...");
    let (perm_balanced_ball, perm_balanced_data) =
        PermutedBall::par_from_ball_tree(balanced_ball.clone(), data.clone(), metric);

    utils::compare_permuted(
        c,
        metric,
        (&ball, &data),
        (&balanced_ball, &perm_balanced_data),
        (&perm_ball, &perm_data),
        (&perm_balanced_ball, &perm_balanced_data),
        None,
        None,
        queries,
        &radii,
        ks,
        true,
    );
}

fn ann_benchmarks(c: &mut Criterion) {
    let root_str = std::env::var("ANN_DATA_ROOT").unwrap();
    println!("ANN data root: {root_str}");
    let ann_data_root = std::path::Path::new(&root_str).canonicalize().unwrap();
    println!("ANN data root: {ann_data_root:?}");

    let seed = Some(42);
    let radii_fractions = vec![0.001, 0.01, 0.1];
    let ks = vec![1, 10, 100];
    let num_queries = 100;

    let (f_mnist, queries) = AnnDataset::FashionMnist.read(&ann_data_root, "npy");
    let queries = &queries[..num_queries];
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        None,
    );
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((2, 1.0)),
    );
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((4, 1.0)),
    );
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((8, 1.0)),
    );
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((16, 1.0)),
    );
    run_search(
        c,
        f_mnist.clone(),
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((32, 1.0)),
    );
    run_search(
        c,
        f_mnist,
        &metric::Euclidean,
        queries,
        &radii_fractions,
        &ks,
        seed,
        Some((64, 1.0)),
    );

    let (sift, queries) = AnnDataset::Sift.read(&ann_data_root, "npy");
    let queries = &queries[..num_queries];
    run_search(c, sift, &metric::Euclidean, queries, &radii_fractions, &ks, seed, None);

    let (glove_25, queries) = AnnDataset::Glove25.read(&ann_data_root, "npy");
    let queries = &queries[..num_queries];
    run_search(c, glove_25, &metric::Cosine, queries, &radii_fractions, &ks, seed, None);
}

criterion_group!(benches, ann_benchmarks);
criterion_main!(benches);
