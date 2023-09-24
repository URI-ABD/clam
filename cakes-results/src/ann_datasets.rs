//! Data sets for the ANN experiments.

/// The data sets to use for the experiments.
///
/// TODO: Add functions to read from hdf5 files.
pub enum AnnDatasets {
    /// The deep-image data set.
    DeepImage,
    /// The fashion-mnist data set.
    FashionMnist,
    /// The gist data set.
    Gist,
    /// The glove-25 data set.
    Glove25,
    /// The glove-50 data set.
    Glove50,
    /// The glove-100 data set.
    Glove100,
    /// The glove-200 data set.
    Glove200,
    /// The mnist data set.
    Mnist,
    /// The sift data set.
    Sift,
    /// The lastfm data set.
    ///
    /// TODO: This data set needs to be reconverted to a numpy file because the current dtype is f64.
    LastFm,
}

impl AnnDatasets {
    /// Return the data set corresponding to the given string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "deep-image" => Ok(Self::DeepImage),
            "fashion-mnist" => Ok(Self::FashionMnist),
            "gist" => Ok(Self::Gist),
            "glove-25" => Ok(Self::Glove25),
            "glove-50" => Ok(Self::Glove50),
            "glove-100" => Ok(Self::Glove100),
            "glove-200" => Ok(Self::Glove200),
            "mnist" => Ok(Self::Mnist),
            "sift" => Ok(Self::Sift),
            "lastfm" => Ok(Self::LastFm),
            _ => Err(format!("Unknown dataset: {s}")),
        }
    }

    /// Return the name of the data set.
    pub const fn name<'a>(&self) -> &'a str {
        match self {
            Self::DeepImage => "deep-image",
            Self::FashionMnist => "fashion-mnist",
            Self::Gist => "gist",
            Self::Glove25 => "glove-25",
            Self::Glove50 => "glove-50",
            Self::Glove100 => "glove-100",
            Self::Glove200 => "glove-200",
            Self::Mnist => "mnist",
            Self::Sift => "sift",
            Self::LastFm => "lastfm",
        }
    }

    /// Return the metric to use for this data set.
    pub fn metric(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self {
            Self::DeepImage
            | Self::Glove25
            | Self::Glove50
            | Self::Glove100
            | Self::Glove200
            | Self::LastFm => distances::vectors::cosine,
            Self::FashionMnist | Self::Gist | Self::Mnist | Self::Sift => {
                distances::simd::euclidean_f32
            }
        }
    }

    /// Read the data set from the given directory.
    pub fn read(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let train_path = dir.join(format!("{}-train.npy", self.name()));
        let train_data = Self::read_npy(&train_path)?;

        let test_path = dir.join(format!("{}-test.npy", self.name()));
        let test_data = Self::read_npy(&test_path)?;

        Ok([train_data, test_data])
    }

    /// Read a numpy file into a vector of vectors.
    fn read_npy(path: &std::path::Path) -> Result<Vec<Vec<f32>>, String> {
        let data: ndarray::Array2<f32> = ndarray_npy::read_npy(path).map_err(|error| {
            format!(
                "Error: Failed to read your dataset at {}. {}",
                path.display(),
                error
            )
        })?;

        Ok(data.outer_iter().map(|row| row.to_vec()).collect())
    }
}
