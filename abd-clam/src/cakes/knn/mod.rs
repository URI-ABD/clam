//! K-Nearest Neighbors (KNN) variants.

mod _knn;
mod knn_thresholds_no_sep_centers;
#[allow(clippy::module_name_repetitions)] // clippy is wrong in this case
pub use _knn::KnnAlgorithm;
