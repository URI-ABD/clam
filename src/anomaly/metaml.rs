use std::path::Path;

use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

use automl::IntoSupervisedData;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;

/// Trait to represent types that can be used as a meta-ML model
pub trait MetaMLModel {
    /// Train the model on the given features and targets
    ///
    /// # Panics
    /// * If the number of columns in the features data isn't 6
    /// * If the number of rows in the features data doesn't match the number of
    /// elements in the targets data
    ///
    /// # Examples
    /// ```
    /// use abd_clam::anomaly::metaml::{MetaMLDataset, MetaMLModel};
    /// use ndarray::{Array1, Array2, arr1, arr2};
    /// use std::path::Path;
    ///
    /// struct ExampleModel {};
    ///
    /// impl ExampleModel {
    ///     pub fn new() -> Self {
    ///         Self {}
    ///     }
    /// }
    ///
    /// impl MetaMLModel for ExampleModel {
    ///     fn train(&mut self, dataset: MetaMLDataset) { }
    ///     fn predict(&self, features: &[f32; 6]) -> f32 { todo!(); }
    ///     fn load(path: &Path) -> Result<Self, String> { todo!(); }
    ///     fn save(&self, path: &Path) -> Result<(), String> { todo!(); }
    /// }
    ///
    /// let n = 10;
    /// let features = vec![[0.; 6]; n];
    /// let targets = vec![0.; n];
    /// let dataset = MetaMLDataset::new(&features, &targets).unwrap();
    ///
    /// // Create and train the model on the dataset
    /// let mut model = ExampleModel::new();
    /// model.train(dataset);
    ///
    /// // Do something with your trained model
    /// ```
    fn train(&mut self, dataset: MetaMLDataset);

    /// Makes a prediction given a trained model and 6 feature values
    ///
    /// # Panics
    /// * If the model hasn't been trained
    ///
    /// # Examples
    /// ```
    /// # use abd_clam::anomaly::metaml::{MetaMLDataset, MetaMLModel};
    /// # use ndarray::{Array1, Array2, arr1, arr2};
    /// # use std::path::Path;
    /// #
    /// # struct ExampleModel {};
    /// #
    /// # impl ExampleModel {
    /// #     pub fn new() -> Self {
    /// #         Self {}
    /// #     }
    /// # }
    /// #
    /// # impl MetaMLModel for ExampleModel {
    /// #     fn train(&mut self, dataset: MetaMLDataset) { }
    /// #     fn predict(&self, features: &[f32; 6]) -> f32 { 0.0 }
    /// #     fn load(path: &Path) -> Result<Self, String> { todo!(); }
    /// #     fn save(&self, path: &Path) -> Result<(), String> { todo!(); }
    /// # }
    /// #
    /// # let n = 10;
    /// # let features = vec![[0.; 6]; n];
    /// # let targets = vec![0.; n];
    /// # let dataset = MetaMLDataset::new(&features, &targets).unwrap();
    /// #
    /// # let mut model = ExampleModel::new();
    /// # model.train(dataset);
    /// #
    /// // This is just random example query.
    /// let query = [4.028, 5.758, 1.402, 0.927, 0.005, 5.502];
    ///
    /// // Model must be trained before using .predict()
    /// // Make a prediction with your model from the query.
    /// let prediction: f32 = model.predict(&query);
    ///
    /// // Do something with the prediction
    /// println!("The model predicted {prediction:.4}");
    /// ```
    fn predict(&self, features: &[f32; 6]) -> f32;

    /// Loads a trained meta-ml model from disk.
    ///
    /// # Error conditions
    /// * If the serialized model cannot be read from the input file path
    /// * If the trained model cannot be deserialized
    ///
    /// # Examples
    /// ```no_run
    /// # // This test is not run for two reasons:
    /// # //   1. Its primary purpose is to read a file from disk
    /// # //   2. The precise format of the file on disk is entirely up to the
    /// # //      implementor of this trait
    /// # use abd_clam::anomaly::metaml::MetaMLDataset;
    /// # use abd_clam::anomaly::metaml::MetaMLModel;
    /// # use std::path::Path;
    /// #
    /// # struct ExampleModel {};
    /// # impl MetaMLModel for ExampleModel {
    /// #     fn train(&mut self, dataset: MetaMLDataset) { todo!(); }
    /// #     fn predict(&self, features: &[f32; 6]) -> f32 { todo!(); }
    /// #     fn load(path: &Path) -> Result<Self, String> { todo!(); }
    /// #     fn save(&self, path: &Path) -> Result<(), String> { todo!(); }
    /// # }
    /// #
    /// // Load a pre-trained model from disk
    /// let path = Path::new("path/to/trained/model.file");
    /// let mut model = ExampleModel::load(&path).unwrap();
    ///
    /// // Do something with your loaded model
    /// ```
    fn load(path: &Path) -> Result<Self, String>
    where
        Self: Sized;

    /// Saves a trained meta-ml model to disk
    ///
    /// # Error conditions
    /// * If the model hasn't been trained
    /// * If the trained model cannot be serialized
    /// * If the serialized model cannot be written to the output file path
    ///
    /// # Examples
    /// ```no_run
    /// # // This test is not run for two reasons:
    /// # //   1. Its primary purpose is to write a file to disk
    /// # //   2. The precise output written to disk is entirely up to the
    /// # //      implementor of this trait
    /// #
    /// # use abd_clam::anomaly::metaml::MetaMLDataset;
    /// # use abd_clam::anomaly::metaml::MetaMLModel;
    /// # use std::path::Path;
    /// #
    /// # struct ExampleModel {};
    /// # impl MetaMLModel for ExampleModel {
    /// #     fn train(&mut self, dataset: MetaMLDataset) { todo!(); }
    /// #     fn predict(&self, features: &[f32; 6]) -> f32 { todo!(); }
    /// #     fn load(path: &Path) -> Result<Self, String> { todo!(); }
    /// #     fn save(&self, path: &Path) -> Result<(), String> { todo!(); }
    /// # }
    /// #
    /// # let dataset: MetaMLDataset = todo!();
    /// # let mut model: ExampleModel = todo!();
    /// # model.train(dataset);
    /// #
    /// // Model must be trained before using .save()
    /// // Save the model to disk for later use
    /// let output_path = Path::new("path/to/output/model.file");
    /// model.save(&output_path);
    /// ```
    fn save(&self, path: &Path) -> Result<(), String>;
}

/// Represents the training data for a MetaML model
///
/// # Invariants:
/// * The number of columns in the features data is 6
/// * The number of rows in the features data is equal to the number of rows in the target data
/// * The data at row `i` of the features data corresponds to the data at row `i` of the targets data
pub struct MetaMLDataset {
    features: DenseMatrix<f32>,
    targets: Vec<f32>,
}

impl MetaMLDataset {
    /// Creates a dataset for training a meta-ml model from a set of feature values
    /// and their corresponding target values
    ///
    /// # Error conditions
    /// * If the number of columns in the features data isn't 6
    /// * If the number of rows in the features data doesn't match the number of
    /// elements in the targets data
    ///
    /// # Examples
    /// ```
    /// use ndarray::{Array1, Array2, arr1, arr2};
    /// use abd_clam::anomaly::metaml::MetaMLDataset;
    ///
    /// let n = 10;
    /// let features = vec![[0.; 6]; n];
    /// let targets = vec![0.; n];
    ///
    /// // Remember features must have the same number of rows as targets
    /// assert!(features.len() == targets.len());
    ///
    /// let dataset = MetaMLDataset::new(&features, &targets).unwrap();
    ///
    /// // Do something with the dataset
    /// ```
    pub fn new(features: &[[f32; 6]], targets: &[f32]) -> Result<Self, String> {
        // TODO: better error checking once the rust branch is merged into master
        if features.len() != targets.len() {
            Err("Different number of features and targets in input data".to_string())
        } else {
            let features = DenseMatrix::from_2d_vec(&features.iter().map(|f| f.to_vec()).collect::<Vec<_>>());
            let targets = targets.to_vec();
            Ok(MetaMLDataset { features, targets })
        }
    }

    /// Creates a dataset for training a meta-ml model from input data on disk.
    ///
    /// # Error conditions
    /// * If either of the given paths can't be converted to a string
    /// * If either of the given files can't be found, opened, or parsed as `f32`s
    /// * If the data contained within the features file isn't two-dimensional
    /// * If the data contained within the targets file isn't one-dimensional
    /// * If the number of columns in the features data isn't 6
    /// * If the number of rows in the features data doesn't match the number of
    /// elements in the targets data
    ///
    /// # Examples
    /// ```
    /// use abd_clam::anomaly::metaml::MetaMLDataset;
    /// use std::path::Path;
    ///
    /// // File paths to training data in the numpy .npy file format
    /// let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let features_path = Path::new(&manifest_dir).join("data/anomaly/dummy_features.npy");
    /// let targets_path = Path::new(&manifest_dir).join("data/anomaly/dummy_targets.npy");
    ///
    /// // Construct a dataset from the features and targets files
    /// let dataset = MetaMLDataset::from_npy(&features_path, &targets_path).unwrap();
    ///
    /// // Do something with the dataset
    /// ```
    pub fn from_npy(features_file_path: &Path, targets_file_path: &Path) -> Result<Self, String> {
        let features_f64: Array2<f64> = read_npy(
            features_file_path
                .to_str()
                .ok_or_else(|| "failed to convert PathBuf to string".to_string())?,
        )
        .map_err(|_| "failed to read the features data file".to_string())?;
        let targets_f64: Array1<f64> = read_npy(
            targets_file_path
                .to_str()
                .ok_or_else(|| "failed to convert PathBuf to string".to_string())?,
        )
        .map_err(|_| "failed to read the outputs data file".to_string())?;

        // Ensure the input data has the correct shape
        if features_f64.ncols() != 6 {
            return Err(format!(
                "Input features had {} columns (expected 6)",
                features_f64.ncols()
            ));
        }
        if features_f64.nrows() != targets_f64.len() {
            return Err(format!(
                "Input features had {} data points, but targets had {}",
                features_f64.nrows(),
                targets_f64.len(),
            ));
        }

        // Transform the training data from f64 to f32 (we are given f64, but automl uses f32s)
        let features: Array2<f32> = features_f64.map(|x| *x as f32);
        let targets: Array1<f32> = targets_f64.map(|x| *x as f32);

        // Transform the training data to vectors. This won't fail, we checked
        // that the data has the correct shape earlier
        let features: Vec<[f32; 6]> = features
            .rows()
            .into_iter()
            .map(|row| row.into_iter().copied().collect::<Vec<_>>().try_into().unwrap())
            .collect();
        let targets: Vec<f32> = targets.to_vec();

        Self::new(&features, &targets)
    }
}

impl IntoSupervisedData for MetaMLDataset {
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        (self.features, self.targets)
    }
}
