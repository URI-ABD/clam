/// CLAM Dataset.
///
/// This module contains the declaration and definition of the Dataset struct.
use std::fmt::Debug;

use ndarray::prelude::*;

use crate::prelude::*;

/// Dataset Trait.
///
/// All datasets supplied to CLAM must implement this trait.
///
pub trait Dataset<T, U>: Debug + Send + Sync {
    /// Returns the name of the metric used to compute the distance between instances.
    ///
    /// :warning: This name must be available in the distances crate.
    fn metric(&self) -> &'static str; // should this return the function directly?

    /// Returns the number of instances in the dataset.
    fn ninstances(&self) -> usize;

    /// Returns the shape of the dataset.
    fn shape(&self) -> Vec<usize>;

    /// Returns the Indices for the dataset.
    fn indices(&self) -> Indices;

    /// Returns the instance at the given index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the instance to return in the dataset.
    ///
    fn instance(&self, index: Index) -> ArrayView<T, IxDyn>;

    /// Selects `n` unique instances from the given indices and returns their indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices to select instances from
    /// * `n` - The number of indices to return
    ///
    fn choose_unique(&self, indices: Indices, n: usize) -> Indices;

    /// Computes the distance between the two instances at the indices provided.
    ///
    /// # Arguments
    ///
    /// * `left` - Index of an instance to compute distance from
    /// * `right`- Index of an instance to compute distance from
    ///
    fn distance(&self, left: Index, right: Index) -> U;

    /// Computes the distances from the instances at left to right.
    ///
    /// # Arguments
    ///
    /// * `left` - Index of the instance to compute distances from
    /// * `right` - Indices of the instances to compute distances to
    #[allow(clippy::ptr_arg)]
    fn distances_from(&self, left: Index, right: &Indices) -> Vec<U>;

    /// Computes and returns distances amongst the instances at left and right.
    ///
    /// # Arguments
    ///
    /// * `left` - Reference to the indices of instances to compute distances among
    /// * `right` - Reference to the indices of instances to compute distances among
    #[allow(clippy::ptr_arg)]
    fn distances_among(&self, left: &Indices, right: &Indices) -> Vec<Vec<U>>;

    /// Computes and returns the pairwise distances between the instances at the given indices.
    ///
    /// # Arguments
    ///
    /// * `indices` - Reference to indices of instances to compute pairwise distances on.
    #[allow(clippy::ptr_arg)]
    fn pairwise_distances(&self, indices: &Indices) -> Vec<Vec<U>>;

    /// Clears the dataset cache.
    fn clear_cache(&self) {}

    /// Returns the size of the cache used for the dataset.
    fn cache_size(&self) -> Option<usize>;
}
