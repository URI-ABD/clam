use automl::IntoSupervisedData;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use std::path::Path;

/// Trait to represent types that can be used as a meta-ML model
pub trait MetaMLModel {
    /// Train the model on the given features and targets.
    ///
    /// This function is used to train the model using the provided dataset, consisting of features and targets.
    ///
    /// # Arguments
    ///
    /// * `dataset`: A `MetaMLDataset` containing the features and targets used for training.
    ///
    /// # Panics
    ///
    /// This function may panic in the following situations:
    ///
    /// * If the number of columns in the features data isn't 6.
    /// * If the number of rows in the features data doesn't match the number of elements in the targets data.
    ///
    fn train(&mut self, dataset: MetaMLDataset);

    /// Makes a prediction given a trained model and 6 feature values.
    ///
    /// This function is used to make a prediction using the provided feature values and a previously
    /// trained model.
    ///
    /// # Arguments
    ///
    /// * `features`: A reference to an array of 6 feature values.
    ///
    /// # Panics
    ///
    /// This function may panic in the following situation:
    ///
    /// * If the model hasn't been trained.
    ///
    fn predict(&self, features: &[f32; 6]) -> f32;

    /// Loads a trained meta-ml model from disk.
    ///
    /// This function is used to load a previously trained meta-ml model from the specified file path.
    ///
    /// # Arguments
    ///
    /// * `path`: A reference to the file path where the model is stored.
    ///
    /// # Errors
    ///
    /// This function can return errors in the following cases:
    ///
    /// * If the serialized model cannot be read from the input file path.
    /// * If the trained model cannot be deserialized.
    ///
    /// # Returns
    ///
    /// If successful, this function returns the loaded meta-ml model.
    ///
    fn load(path: &Path) -> Result<Self, String>
    where
        Self: Sized;

    /// Saves a trained meta-ml model to disk.
    ///
    /// # Arguments
    ///
    /// * `path`: A reference to the file path where the model will be saved.
    ///
    /// # Returns
    ///
    /// Returns `Result<(), String>` where `Ok(())` indicates success, and `Err` contains an error message.
    ///
    /// # Errors
    /// * If the model hasn't been trained.
    /// * If the trained model cannot be serialized.
    /// * If the serialized model cannot be written to the output file path.
    ///
    fn save(&self, path: &Path) -> Result<(), String>;
}

/// Represents the training data for a `MetaML` model
///
/// # Invariants:
/// * The number of columns in the features data is 6
/// * The number of rows in the features data is equal to the number of rows in the target data
/// * The data at row `i` of the features data corresponds to the data at row `i` of the targets data
pub struct MetaMLDataset {
    /// Features data for training the `MetaML` model.
    features: DenseMatrix<f32>,
    /// Target values for the corresponding features data.
    targets: Vec<f32>,
}

impl MetaMLDataset {
    /// Creates a dataset for training a meta-ml model from a set of feature values and their corresponding target values.
    ///
    /// # Arguments
    ///
    /// * `_features`: A slice of arrays representing feature values, where each array has 6 elements.
    /// * `_targets`: A slice of target values.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Self, String>` where `Ok(Self)` indicates success, and `Err` contains an error message.
    ///
    /// # Errors
    /// * If the number of columns in the features data isn't 6.
    /// * If the number of rows in the features data doesn't match the number of elements in the targets data.
    ///
    pub fn new(_features: &[[f32; 6]], _targets: &[f32]) -> Result<Self, String> {
        todo!()
        // TODO: better error checking once the rust branch is merged into master
        // if features.len() == targets.len() {
        //     Err("Different number of features and targets in input data".to_string())
        // } else {
        //     let features = DenseMatrix::from_2d_vec(&features.iter().map(|f| f.to_vec()).collect::<Vec<_>>());
        //     let targets = targets.to_vec();
        //     Ok(Self { features, targets })
        // }
    }

    /// Creates a dataset for training a meta-ml model from input data on disk.
    ///
    /// # Arguments
    ///
    /// * `_features_file_path`: Path to the file containing feature data.
    /// * `_targets_file_path`: Path to the file containing target data.
    ///
    /// # Returns
    ///
    /// Returns a `Result<Self, String>` where `Ok(Self)` indicates success, and `Err` contains an error message.
    ///
    /// # Errors
    /// * If either of the given paths can't be converted to a string.
    /// * If either of the given files can't be found, opened, or parsed as `f32`s.
    /// * If the data contained within the features file isn't two-dimensional.
    /// * If the data contained within the targets file isn't one-dimensional.
    /// * If the number of columns in the features data isn't 6.
    /// * If the number of rows in the features data doesn't match the number of elements in the targets data.
    ///
    pub fn from_npy(_features_file_path: &Path, _targets_file_path: &Path) -> Result<Self, String> {
        todo!()

        // let features_f64: Array2<f64> = read_npy(
        //     features_file_path
        //         .to_str()
        //         .ok_or_else(|| "failed to convert PathBuf to string".to_string())?,
        // )
        // .map_err(|_| "failed to read the features data file".to_string())?;
        // let targets_f64: Array1<f64> = read_npy(
        //     targets_file_path
        //         .to_str()
        //         .ok_or_else(|| "failed to convert PathBuf to string".to_string())?,
        // )
        // .map_err(|_| "failed to read the outputs data file".to_string())?;
        //
        // // Ensure the input data has the correct shape
        // if features_f64.ncols() != 6 {
        //     return Err(format!(
        //         "Input features had {} columns (expected 6)",
        //         features_f64.ncols()
        //     ));
        // }
        // if features_f64.nrows() != targets_f64.len() {
        //     return Err(format!(
        //         "Input features had {} data points, but targets had {}",
        //         features_f64.nrows(),
        //         targets_f64.len(),
        //     ));
        // }
        //
        // // Transform the training data from f64 to f32 (we are given f64, but automl uses f32s)
        // let features: Array2<f32> = features_f64.map(|x| *x as f32);
        // let targets: Array1<f32> = targets_f64.map(|x| *x as f32);
        //
        // // Transform the training data to vectors. This won't fail, we checked
        // // that the data has the correct shape earlier
        // let features: Vec<[f32; 6]> = features
        //     .rows()
        //     .into_iter()
        //     .map(|row| row.into_iter().copied().collect::<Vec<_>>().try_into().unwrap())
        //     .collect();
        // let targets: Vec<f32> = targets.to_vec();
        //
        // Self::new(&features, &targets)
    }
}

impl IntoSupervisedData for MetaMLDataset {
    /// Converts the current dataset into a tuple containing feature data and target data.
    ///
    /// This function transforms the dataset into a format suitable for supervised learning, returning
    /// a tuple where the first element is a two-dimensional feature matrix, and the second element is
    /// a one-dimensional target vector.
    ///
    /// # Returns
    ///
    /// A tuple containing feature data represented as a `DenseMatrix<f32>` and target data represented
    /// as a `Vec<f32>`.
    fn to_supervised_data(self) -> (DenseMatrix<f32>, Vec<f32>) {
        (self.features, self.targets)
    }
}
