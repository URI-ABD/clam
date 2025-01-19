//! Utilities for reading the CHAODA datasets.

use std::path::Path;

use abd_clam::{dataset::AssociatesMetadataMut, Dataset, FlatVec};
use ndarray::prelude::*;
use ndarray_npy::ReadNpyExt;

type ChaodaDataset = FlatVec<Vec<f64>, bool>;

/// The datasets used for anomaly detection.
///
/// These are taken from <https://odds.cs.stonybrook.edu>
pub enum Data {
    /// 7200 x 6, 534 outliers
    Annthyroid,
    /// 452 x 274, 66 outliers
    Arrhythmia,
    /// 683 x 9, 239 outliers
    BreastW,
    /// 1831 x 21, 176 outliers
    Cardio,
    /// 286048 x 10, 2747 outliers
    ForestCover,
    /// 214 x 9, 9 outliers
    Glass,
    /// 567479 x 3, 2211 outliers
    Http,
    /// 351 x 33, 126 outliers
    Ionosphere,
    /// 148 x 18, 6 outliers
    Lympho,
    /// 11183 x 6, 260 outliers
    Mammography,
    /// 7603 x 100, 700 outliers
    Mnist,
    /// 3062 x 166, 97 outliers
    Musk,
    /// 5216 x 64, 150 outliers
    OptDigits,
    /// 6870 x 16, 156 outliers
    PenDigits,
    /// 768 x 8, 268 outliers
    Pima,
    /// 6435 x 36, 2036 outliers
    Satellite,
    /// 5803 x 36, 71 outliers
    SatImage2,
    /// 49097 x 9, 3511 outliers
    Shuttle,
    /// 95156 x 3, 30 outliers
    Smtp,
    /// 3772 x 6, 93 outliers
    Thyroid,
    /// 240 x 6, 30 outliers
    Vertebral,
    /// 1456 x 12, 50 outliers
    Vowels,
    /// 278 x 30, 21 outliers
    Wbc,
    /// 129 x 13, 10 outliers
    Wine,
}

impl Data {
    /// Create a new dataset from the name.
    #[allow(dead_code)]
    pub fn new(name: &str) -> Result<Self, String> {
        match name {
            "annthyroid" => Ok(Self::Annthyroid),
            "arrhythmia" => Ok(Self::Arrhythmia),
            "breastw" => Ok(Self::BreastW),
            "cardio" => Ok(Self::Cardio),
            "cover" => Ok(Self::ForestCover),
            "glass" => Ok(Self::Glass),
            "http" => Ok(Self::Http),
            "ionosphere" => Ok(Self::Ionosphere),
            "lympho" => Ok(Self::Lympho),
            "mammography" => Ok(Self::Mammography),
            "mnist" => Ok(Self::Mnist),
            "musk" => Ok(Self::Musk),
            "optdigits" => Ok(Self::OptDigits),
            "pendigits" => Ok(Self::PenDigits),
            "pima" => Ok(Self::Pima),
            "satellite" => Ok(Self::Satellite),
            "satimage-2" => Ok(Self::SatImage2),
            "shuttle" => Ok(Self::Shuttle),
            "smtp" => Ok(Self::Smtp),
            "thyroid" => Ok(Self::Thyroid),
            "vertebral" => Ok(Self::Vertebral),
            "vowels" => Ok(Self::Vowels),
            "wbc" => Ok(Self::Wbc),
            "wine" => Ok(Self::Wine),
            _ => Err(format!("Unknown dataset: {name}")),
        }
    }

    /// Get the name of the dataset.
    ///
    /// The dataset name is the name of the file without the extension.
    #[must_use]
    pub const fn name(&self) -> &str {
        match self {
            Self::Annthyroid => "annthyroid",
            Self::Arrhythmia => "arrhythmia",
            Self::BreastW => "breastw",
            Self::Cardio => "cardio",
            Self::ForestCover => "cover",
            Self::Glass => "glass",
            Self::Http => "http",
            Self::Ionosphere => "ionosphere",
            Self::Lympho => "lympho",
            Self::Mammography => "mammography",
            Self::Mnist => "mnist",
            Self::Musk => "musk",
            Self::OptDigits => "optdigits",
            Self::PenDigits => "pendigits",
            Self::Pima => "pima",
            Self::Satellite => "satellite",
            Self::SatImage2 => "satimage-2",
            Self::Shuttle => "shuttle",
            Self::Smtp => "smtp",
            Self::Thyroid => "thyroid",
            Self::Vertebral => "vertebral",
            Self::Vowels => "vowels",
            Self::Wbc => "wbc",
            Self::Wine => "wine",
        }
    }

    /// Read the dataset.
    ///
    /// # Arguments
    ///
    /// * `data_dir` - The directory containing the dataset.
    pub fn read(&self, data_dir: &Path) -> Result<ChaodaDataset, String> {
        match self {
            Self::Annthyroid
            | Self::Arrhythmia
            | Self::BreastW
            | Self::Cardio
            | Self::ForestCover
            | Self::Glass
            | Self::Http
            | Self::Ionosphere
            | Self::Lympho
            | Self::Mammography
            | Self::Mnist
            | Self::Musk
            | Self::OptDigits
            | Self::PenDigits
            | Self::Pima
            | Self::Satellite
            | Self::SatImage2
            | Self::Shuttle
            | Self::Smtp
            | Self::Thyroid
            | Self::Vertebral
            | Self::Vowels
            | Self::Wbc
            | Self::Wine => read_xy(data_dir, self.name()),
        }
    }

    /// Read the training datasets from the paper
    pub fn read_train_data(data_dir: &Path) -> Result<[ChaodaDataset; 4], String> {
        Ok([
            Self::Annthyroid.read(data_dir)?,
            // Self::Mnist.read(data_dir)?,
            Self::PenDigits.read(data_dir)?,
            Self::Satellite.read(data_dir)?,
            // Self::Shuttle.read(data_dir)?,  // It takes too long to train
            Self::Thyroid.read(data_dir)?,
        ])
    }

    /// Read the inference datasets from the paper
    pub fn read_infer_data(data_dir: &Path) -> Result<Vec<ChaodaDataset>, String> {
        Ok(vec![
            Self::Arrhythmia.read(data_dir)?,
            Self::BreastW.read(data_dir)?,
            Self::Cardio.read(data_dir)?,
            Self::ForestCover.read(data_dir)?,
            Self::Glass.read(data_dir)?,
            Self::Http.read(data_dir)?,
            Self::Ionosphere.read(data_dir)?,
            Self::Lympho.read(data_dir)?,
            Self::Mammography.read(data_dir)?,
            Self::Musk.read(data_dir)?,
            Self::OptDigits.read(data_dir)?,
            Self::Pima.read(data_dir)?,
            Self::SatImage2.read(data_dir)?,
            Self::Smtp.read(data_dir)?,
            Self::Vertebral.read(data_dir)?,
            Self::Vowels.read(data_dir)?,
            Self::Wbc.read(data_dir)?,
            Self::Wine.read(data_dir)?,
        ])
    }

    /// Read all the datasets
    pub fn read_all(data_dir: &Path) -> Result<Vec<ChaodaDataset>, String> {
        let mut datasets = Self::read_train_data(data_dir)?.to_vec();
        datasets.extend(Self::read_infer_data(data_dir)?);
        Ok(datasets)
    }
}

fn read_xy(path: &Path, name: &str) -> Result<ChaodaDataset, String> {
    let labels_path = path.join(format!("{name}_labels.npy"));
    let reader = std::fs::File::open(labels_path).map_err(|e| format!("Could not open file: {e}"))?;
    let labels = Array1::<u8>::read_npy(reader).map_err(|e| format!("Could not read labels from file {path:?}: {e}"))?;
    let labels = labels.mapv(|y| y == 1).to_vec();

    let x_path = path.join(format!("{name}.npy"));
    let fv: FlatVec<Vec<f64>, usize> = FlatVec::read_npy(&x_path)?;
    fv.with_name(name).with_metadata(&labels)
}
