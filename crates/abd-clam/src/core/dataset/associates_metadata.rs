//! An extension of the `Dataset` trait that provides methods for working with
//! metadata associated with items in a dataset.

use super::Dataset;

/// A trait that extends the `Dataset` trait with methods for working with
/// metadata associated with items in a dataset.
///
/// Each item in the dataset should be associated with a piece of metadata.
///
/// # Type parameters
///
/// - `I`: The items in the dataset.
/// - `Me`: The metadata associated with each item in the dataset.
pub trait AssociatesMetadata<I, Me>: Dataset<I> {
    /// Returns the all metadata associated with the items in the dataset.
    fn metadata(&self) -> &[Me];

    /// Returns the metadata associated with the item at the given index.
    fn metadata_at(&self, index: usize) -> &Me;
}

/// An extension of the `AssociatesMetadata` trait that provides methods for
/// changing the metadata associated with items in a dataset.
///
/// # Type parameters
///
/// - `I`: The items in the dataset.
/// - `Me`: The metadata associated with each item in the dataset.
/// - `Met`: The metadata that can be associated after transformation.
/// - `D`: The type of the dataset after the transformation.
#[allow(clippy::module_name_repetitions)]
pub trait AssociatesMetadataMut<I, Me, Met: Clone, D: AssociatesMetadata<I, Met>>: AssociatesMetadata<I, Me> {
    /// Returns the all metadata associated with the items in the dataset,
    /// mutably.
    fn metadata_mut(&mut self) -> &mut [Me];

    /// Returns the metadata associated with the item at the given index,
    /// mutably.
    fn metadata_at_mut(&mut self, index: usize) -> &mut Me;

    /// Changes all metadata associated with the items in the dataset.
    ///
    /// # Errors
    ///
    /// - If the number of metadata items is not equal to the cardinality of the
    ///   dataset.
    fn with_metadata(self, metadata: &[Met]) -> Result<D, String>;

    /// Applies a transformation to the metadata associated with the items in
    /// the dataset.
    fn transform_metadata<F: Fn(&Me) -> Met>(self, f: F) -> D;
}
