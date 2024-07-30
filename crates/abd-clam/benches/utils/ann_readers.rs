//! Utilities for reading the datasets we downloaded from the `ann-benchmarks`
//! repository.

use std::{
    ffi::OsStr,
    path::{Path, PathBuf},
};

use abd_clam::{FlatVec, Metric};

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
        let (metric, test, _, _, _) = test.deconstruct();
        let train = FlatVec::read_npy(train, metric).unwrap();
        (train, test)
    }
}
