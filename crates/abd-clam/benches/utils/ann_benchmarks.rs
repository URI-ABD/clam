//! Utilities for the ANN-Benchmarks datasets.
//!
//! Their data can be found [here](https://github.com/erikbern/ann-benchmarks).

use std::path::{Path, PathBuf};

use rand::prelude::*;

use super::metrics;

#[derive(Debug)]
#[allow(dead_code)]
pub enum AnnDataset {
    // Euclidean
    FashionMnist,
    Mnist,
    Sift,
    Gist,
    // Cosine
    Glove25,
    Glove50,
    Glove100,
    Glove200,
    DeepImage,
}

impl AnnDataset {
    pub fn name(&self) -> &'static str {
        match self {
            Self::FashionMnist => "fmnist-784",
            Self::Mnist => "mnist-784",
            Self::Sift => "sift-128",
            Self::Gist => "gist-960",
            Self::Glove25 => "glove-25",
            Self::Glove50 => "glove-50",
            Self::Glove100 => "glove-100",
            Self::Glove200 => "glove-200",
            Self::DeepImage => "deepimage-96",
        }
    }

    pub fn metric<I: AsRef<[f32]>>(&self) -> fn(&I, &I) -> f32 {
        match self {
            Self::FashionMnist | Self::Mnist | Self::Sift | Self::Gist => metrics::euclidean,
            Self::Glove25 | Self::Glove50 | Self::Glove100 | Self::Glove200 | Self::DeepImage => metrics::cosine,
        }
    }

    fn file_name_prefix(&self) -> &'static str {
        match self {
            Self::FashionMnist => "fashion-mnist",
            Self::Mnist => "mnist",
            Self::Sift => "sift",
            Self::Gist => "gist",
            Self::Glove25 => "glove-25",
            Self::Glove50 => "glove-50",
            Self::Glove100 => "glove-100",
            Self::Glove200 => "glove-200",
            Self::DeepImage => "deep-image",
        }
    }

    pub fn train_path<P: AsRef<Path>>(&self, base: &P) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.subset_path(base, "Train")
    }

    pub fn read_train<P: AsRef<Path>, R: rand::Rng>(&self, base: &P, rng: Option<&mut R>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let path = self.train_path(base)?;
        let arr = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        let mut vec = array2_to_vec_f32(&arr);
        if let Some(rng) = rng {
            vec.shuffle(rng);
        }
        Ok(vec)
    }

    pub fn test_path<P: AsRef<Path>>(&self, base: &P) -> Result<PathBuf, Box<dyn std::error::Error>> {
        self.subset_path(base, "Test")
    }

    pub fn read_test<P: AsRef<Path>, R: rand::Rng>(&self, base: &P, rng: Option<&mut R>) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        let path = self.test_path(base)?;
        let arr = ndarray_npy::read_npy::<_, ndarray::Array2<f32>>(path).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        let mut vec = array2_to_vec_f32(&arr);
        if let Some(rng) = rng {
            vec.shuffle(rng);
        }
        Ok(vec)
    }

    fn subset_path<P: AsRef<Path>>(&self, base: &P, subset: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let path = base.as_ref().join(format!("{}-{}.npy", self.file_name_prefix(), subset.to_ascii_lowercase()));
        if path.exists() {
            Ok(path)
        } else {
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{subset} subset not found: {path:?}"),
            )))
        }
    }
}

pub fn base_dir() -> Result<PathBuf, String> {
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .map(|p| p.join("../../../data/ann_data"))
        .map_err(|e| format!("Failed to get workspace directory: {e}"))?;

    let base = std::env::var("ANN_BENCHMARKS_DIR").map(PathBuf::from).unwrap_or_else(|_| workspace_dir);

    base.canonicalize()
        .map_err(|e| format!("Failed to canonicalize base directory: {e} ({base:?})"))
}

pub fn array2_to_vec_f32(arr: &ndarray::Array2<f32>) -> Vec<Vec<f32>> {
    arr.outer_iter().map(|row| row.to_vec()).collect()
}
