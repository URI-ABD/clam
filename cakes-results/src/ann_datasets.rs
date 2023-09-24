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
    LastFm,
    /// The NYTimes data set.
    NYTimes,
    /// The kosarak data set.
    Kosarak,
    /// The MovieLens10m data set.
    MovieLens10m,
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
            "nytimes" => Ok(Self::NYTimes),
            "kosarak" => Ok(Self::Kosarak),
            "movielens10m" => Ok(Self::MovieLens10m),
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
            Self::NYTimes => "nytimes",
            Self::Kosarak => "kosarak",
            Self::MovieLens10m => "movielens10m",
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
            | Self::LastFm
            | Self::NYTimes => distances::vectors::cosine,
            Self::FashionMnist | Self::Gist | Self::Mnist | Self::Sift => {
                distances::simd::euclidean_f32
            }
            Self::Kosarak | Self::MovieLens10m => {
                unimplemented!("We are still merging Jaccard distance. Generic distances are hard.")
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

    /// The link from which to download the data set.
    const fn download_link<'a>(&self) -> &'a str {
        match self {
            Self::DeepImage => "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
            Self::FashionMnist => "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
            Self::Gist => "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
            Self::Glove25 => "http://ann-benchmarks.com/glove-25-angular.hdf5",
            Self::Glove50 => "http://ann-benchmarks.com/glove-50-angular.hdf5",
            Self::Glove100 => "http://ann-benchmarks.com/glove-100-angular.hdf5",
            Self::Glove200 => "http://ann-benchmarks.com/glove-200-angular.hdf5",
            Self::Kosarak => "http://ann-benchmarks.com/kosarak-jaccard.hdf5",
            Self::Mnist => "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
            Self::MovieLens10m => "http://ann-benchmarks.com/movielens10m-jaccard.hdf5",
            Self::NYTimes => "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
            Self::Sift => "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
            Self::LastFm => "http://ann-benchmarks.com/lastfm-64-dot.hdf5",
        }
    }

    /// The name of the hdf5 file.
    #[allow(dead_code)]
    fn hdf5_name<'a>(&self) -> &'a str {
        self.download_link().split(".com/").collect::<Vec<_>>()[1]
    }

    /// Read the data set from the given directory.
    #[allow(dead_code, unused_variables)]
    pub fn read_hdf5(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let data_path = dir.join(self.hdf5_name());
        // let file = hdf5::File::open(data_path).map_err(|error| {
        //     format!(
        //         "Error: Failed to read your dataset at {}. {}",
        //         data_path.display(),
        //         error
        //     )
        // })?;

        todo!()
    }
}
