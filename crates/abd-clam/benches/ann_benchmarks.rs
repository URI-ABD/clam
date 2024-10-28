//! Benchmarks for the suite of ANN-Benchmarks datasets.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use abd_clam::{
    adapter::{Adapter, ParBallAdapter},
    cakes::OffBall,
    partition::ParPartition,
    BalancedBall, Ball, Cluster, Dataset, FlatVec, Metric, Permutable,
};
use criterion::*;

mod utils;

/// Reads the training and query data of the given dataset from the directory.
pub fn read_ann_data_npy(
    name: &str,
    root: &Path,
    metric: Metric<Vec<f32>, f32>,
) -> (FlatVec<Vec<f32>, f32, usize>, Vec<Vec<f32>>) {
    AnnDataset::from_name(name).read(root, "npy", metric)
}

/// The datasets available in the `ann-benchmarks` repository.
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
    fn paths<P: AsRef<Path>, S: AsRef<OsStr>>(&self, root: P, ext: S) -> [PathBuf; 2] {
        let name = self.to_name();
        [
            root.as_ref()
                .join(format!("{}-train", name))
                .with_extension(ext.as_ref()),
            root.as_ref().join(format!("{}-test", name)).with_extension(ext),
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
    fn read<P: AsRef<Path>, S: AsRef<OsStr>>(
        &self,
        root: P,
        ext: S,
        metric: Metric<Vec<f32>, f32>,
    ) -> (FlatVec<Vec<f32>, f32, usize>, Vec<Vec<f32>>) {
        let [train, test] = self.paths(root, ext);
        println!("Reading train data from {:?}, {:?}", train, test);
        let test = FlatVec::read_npy(test, metric).unwrap();
        let (metric, test, _, _, _, _) = test.deconstruct();
        let train = FlatVec::read_npy(train, metric).unwrap().with_name(self.to_name());
        (train, test)
    }
}

fn ann_benchmarks(c: &mut Criterion) {
    let root_str = std::env::var("ANN_DATA_ROOT").unwrap();
    let ann_data_root = std::path::Path::new(&root_str).canonicalize().unwrap();
    println!("ANN data root: {:?}", ann_data_root);

    let euclidean = |x: &Vec<_>, y: &Vec<_>| distances::vectors::euclidean(x, y);
    let cosine = |x: &Vec<_>, y: &Vec<_>| distances::vectors::cosine(x, y);

    #[allow(clippy::type_complexity)]
    let data_names: Vec<(&str, &str, Metric<Vec<f32>, f32>)> = vec![
        ("fashion-mnist", "euclidean", Metric::new(euclidean, false)),
        ("glove-25", "cosine", Metric::new(cosine, false)),
        ("sift", "euclidean", Metric::new(euclidean, false)),
    ];

    let data_pairs = data_names.into_iter().map(|(data_name, metric_name, metric)| {
        (
            data_name,
            metric_name,
            read_ann_data_npy(data_name, &ann_data_root, metric),
        )
    });

    let seed = Some(42);
    let radii = vec![];
    let ks = vec![10, 100];
    let num_queries = 100;
    for (data_name, metric_name, (data, queries)) in data_pairs {
        let queries = &queries[0..num_queries];

        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let ball = Ball::par_new_tree(&data, &criteria, seed);
        let (off_ball, perm_data) = OffBall::par_from_ball_tree(ball.clone(), data.clone());

        let criteria = |c: &BalancedBall<_, _, _>| c.cardinality() > 1;
        let balanced_ball = BalancedBall::par_new_tree(&data, &criteria, seed);
        let (balanced_off_ball, balanced_perm_data) = {
            let balanced_off_ball = OffBall::adapt_tree(balanced_ball.clone(), None, &data);
            let mut balanced_perm_data = data.clone();
            let permutation = balanced_off_ball.source().indices().collect::<Vec<_>>();
            balanced_perm_data.permute(&permutation);
            (balanced_off_ball, balanced_perm_data)
        };

        utils::compare_permuted(
            c,
            data_name,
            metric_name,
            (&ball, &data),
            (&off_ball, &perm_data),
            None,
            (&balanced_ball, &data),
            (&balanced_off_ball, &balanced_perm_data),
            None,
            queries,
            &radii,
            &ks,
            true,
        );
    }
}

criterion_group!(benches, ann_benchmarks);
criterion_main!(benches);
