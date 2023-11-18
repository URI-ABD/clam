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
    /// A random data set. (dimensionality, metric_name)
    Random(usize, String),
}

impl AnnDatasets {
    /// Return the data set corresponding to the given string.
    pub fn from_str(s: &str) -> Result<Self, String> {
        let s = s.to_lowercase();

        if s.starts_with("random") {
            let parts = s.split('-').collect::<Vec<_>>();
            if parts.len() != 3 {
                return Err(format!("Unknown dataset: {s}"));
            }
            let dimensionality = parts[1].parse::<usize>().map_err(|reason| {
                format!("Failed to parse dimensionality from random dataset: {s} ({reason})")
            })?;
            let metric_name = parts[2].to_string();
            let dataset = Self::Random(dimensionality, metric_name);
            let _ = dataset.metric()?;
            Ok(dataset)
        } else {
            match s.as_str() {
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
    }

    /// Return the name of the data set.
    pub const fn name(&self) -> &str {
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
            Self::Random(..) => "random",
        }
    }

    /// Return the metric to use for this data set.
    pub fn metric_name(&self) -> &str {
        match self {
            Self::DeepImage
            | Self::Glove25
            | Self::Glove50
            | Self::Glove100
            | Self::Glove200
            | Self::LastFm
            | Self::NYTimes => "cosine",
            Self::FashionMnist | Self::Gist | Self::Mnist | Self::Sift => "euclidean",
            Self::Kosarak | Self::MovieLens10m => "jaccard",
            Self::Random(_, metric_name) => metric_name,
        }
    }

    /// Return the metric to use for this data set.
    #[allow(clippy::type_complexity)]
    pub fn metric(&self) -> Result<fn(&Vec<f32>, &Vec<f32>) -> f32, String> {
        match self.metric_name() {
            "cosine" => Ok(cosine),
            "euclidean" => Ok(euclidean),
            "jaccard" => Err(
                "We are still merging Jaccard distance. Generic distances are hard.".to_string(),
            ),
            _ => Err(format!("Unknown metric: {}", self.metric_name())),
        }
    }

    /// Read the data set from the given directory.
    pub fn read(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let [train_data, test_data] = if let Self::Random(d, _) = self {
            let dimensionality = *d;
            let (min_val, max_val) = (-1.0, 1.0);
            let seed = 42;

            let train_data = symagen::random_data::random_tabular_seedable(
                1_000_000,
                dimensionality,
                min_val,
                max_val,
                seed,
            );
            let test_data = symagen::random_data::random_tabular_seedable(
                10_000,
                dimensionality,
                min_val,
                max_val,
                seed + 1,
            );

            [train_data, test_data]
        } else {
            let train_path = dir.join(format!("{}-train.npy", self.name()));
            let train_data = Self::read_npy(&train_path)?;

            let test_path = dir.join(format!("{}-test.npy", self.name()));
            let test_data = Self::read_npy(&test_path)?;

            [train_data, test_data]
        };

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
    fn download_link<'a>(&self) -> Result<&'a str, String> {
        match self {
            Self::DeepImage => Ok("http://ann-benchmarks.com/deep-image-96-angular.hdf5"),
            Self::FashionMnist => Ok("http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"),
            Self::Gist => Ok("http://ann-benchmarks.com/gist-960-euclidean.hdf5"),
            Self::Glove25 => Ok("http://ann-benchmarks.com/glove-25-angular.hdf5"),
            Self::Glove50 => Ok("http://ann-benchmarks.com/glove-50-angular.hdf5"),
            Self::Glove100 => Ok("http://ann-benchmarks.com/glove-100-angular.hdf5"),
            Self::Glove200 => Ok("http://ann-benchmarks.com/glove-200-angular.hdf5"),
            Self::Kosarak => Ok("http://ann-benchmarks.com/kosarak-jaccard.hdf5"),
            Self::Mnist => Ok("http://ann-benchmarks.com/mnist-784-euclidean.hdf5"),
            Self::MovieLens10m => Ok("http://ann-benchmarks.com/movielens10m-jaccard.hdf5"),
            Self::NYTimes => Ok("http://ann-benchmarks.com/nytimes-256-angular.hdf5"),
            Self::Sift => Ok("http://ann-benchmarks.com/sift-128-euclidean.hdf5"),
            Self::LastFm => Ok("http://ann-benchmarks.com/lastfm-64-dot.hdf5"),
            Self::Random(..) => Err("Random datasets cannot be downloaded.".to_string()),
        }
    }

    /// The name of the hdf5 file.
    #[allow(dead_code)]
    fn hdf5_name<'a>(&self) -> Result<&'a str, String> {
        self.download_link()
            .map(|link| link.split(".com/").collect::<Vec<_>>()[1])
    }

    /// Read the data set from the given directory.
    #[allow(dead_code, unused_variables)]
    pub fn read_hdf5(&self, dir: &std::path::Path) -> Result<[Vec<Vec<f32>>; 2], String> {
        let data_path = dir.join(self.hdf5_name()?);
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

/// A wrapper around the cosine distance function.
#[allow(clippy::ptr_arg)]
fn cosine(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::cosine_f32(x, y)
}

/// A wrapper around the euclidean distance function.
#[allow(clippy::ptr_arg)]
fn euclidean(x: &Vec<f32>, y: &Vec<f32>) -> f32 {
    distances::simd::euclidean_f32(x, y)
}
