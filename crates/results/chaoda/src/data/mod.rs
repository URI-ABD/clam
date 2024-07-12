//! Utilities for reading the CHAODA datasets.

use std::path::Path;

use distances::Number;
use ndarray::prelude::*;
use ndarray_npy::{ReadNpyExt, ReadableElement};

/// The result of reading a dataset.
///
/// The first element is the data and the second element is the labels.
pub type DataResult = (Vec<Vec<f32>>, Vec<bool>);

/// The datasets used for anomaly detection.
///
/// These are taken from https://odds.cs.stonybrook.edu/
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
    pub fn name(&self) -> &str {
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
    pub fn read(&self, data_dir: &Path) -> Result<DataResult, String> {
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
            | Self::Wine => read_xy::<f64, u8>(data_dir, self.name()),
        }
    }

    /// Read the training datasets from the paper
    pub fn read_paper_train(data_dir: &Path) -> Result<Vec<(String, DataResult)>, String> {
        Ok(vec![
            ("annthyroid".to_string(), Self::Annthyroid.read(data_dir)?),
            ("mnist".to_string(), Self::Mnist.read(data_dir)?),
            ("pendigits".to_string(), Self::PenDigits.read(data_dir)?),
            ("satellite".to_string(), Self::Satellite.read(data_dir)?),
            // ("shuttle".to_string(), Self::Shuttle.read(data_dir)?),  // I moved this to inference because it takes too long to train
            ("thyroid".to_string(), Self::Thyroid.read(data_dir)?),
        ])
    }

    /// Read the inference datasets from the paper
    pub fn read_paper_inference(data_dir: &Path) -> Result<Vec<(String, DataResult)>, String> {
        Ok(vec![
            ("shuttle".to_string(), Self::Shuttle.read(data_dir)?),
            ("arrhythmia".to_string(), Self::Arrhythmia.read(data_dir)?),
            ("breastw".to_string(), Self::BreastW.read(data_dir)?),
            ("cardio".to_string(), Self::Cardio.read(data_dir)?),
            ("cover".to_string(), Self::ForestCover.read(data_dir)?),
            ("glass".to_string(), Self::Glass.read(data_dir)?),
            ("http".to_string(), Self::Http.read(data_dir)?),
            ("ionosphere".to_string(), Self::Ionosphere.read(data_dir)?),
            ("lympho".to_string(), Self::Lympho.read(data_dir)?),
            ("mammography".to_string(), Self::Mammography.read(data_dir)?),
            ("musk".to_string(), Self::Musk.read(data_dir)?),
            ("optdigits".to_string(), Self::OptDigits.read(data_dir)?),
            ("pima".to_string(), Self::Pima.read(data_dir)?),
            ("satimage-2".to_string(), Self::SatImage2.read(data_dir)?),
            ("smtp".to_string(), Self::Smtp.read(data_dir)?),
            ("vertebral".to_string(), Self::Vertebral.read(data_dir)?),
            ("vowels".to_string(), Self::Vowels.read(data_dir)?),
            ("wbc".to_string(), Self::Wbc.read(data_dir)?),
            ("wine".to_string(), Self::Wine.read(data_dir)?),
        ])
    }

    /// Read all the datasets
    #[allow(dead_code)]
    pub fn read_all(data_dir: &Path) -> Result<Vec<(String, DataResult)>, String> {
        let mut datasets = Self::read_paper_train(data_dir)?;
        datasets.extend(Self::read_paper_inference(data_dir)?);
        Ok(datasets)
    }
}

fn read_xy<X, Y>(path: &Path, name: &str) -> Result<DataResult, String>
where
    X: Number + ReadableElement,
    Y: Number + ReadableElement,
{
    let x_path = path.join(format!("{}.npy", name));

    let reader = std::fs::File::open(x_path).map_err(|e| e.to_string())?;
    let x_data = Array2::<X>::read_npy(reader).map_err(|e| e.to_string())?;
    let x_data = x_data.mapv(|x| x.as_f32());
    let x_data = x_data.axis_iter(Axis(0)).map(|row| row.to_vec()).collect::<Vec<_>>();

    let y_path = path.join(format!("{}_labels.npy", name));
    let reader = std::fs::File::open(y_path).map_err(|e| e.to_string())?;
    let y_data = Array1::<Y>::read_npy(reader).map_err(|e| e.to_string())?;
    let y_data = y_data.mapv(|y| y == Y::one()).to_vec();

    Ok((x_data, y_data))
}
