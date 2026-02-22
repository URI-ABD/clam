//! A collection of items and their identifiers for use in a `Tree`.

/// A collection of items and their identifiers for use in a `Tree`.
pub trait Dataset: Sized {
    /// The type of the identifier for each item in the dataset.
    ///
    /// This is typically some sort of metadata for its associated item.
    type Id;

    /// The type of items in the dataset.
    ///
    /// This is the actual data stored in the dataset. We build the tree over these items and we can compute distances between them using a provided metric.
    type Item;

    /// Get the items and their identifiers in a slice.
    fn as_slice(&self) -> &[(Self::Id, Self::Item)];

    /// Get the items and their identifiers as a mutable slice.
    fn as_mut_slice(&mut self) -> &mut [(Self::Id, Self::Item)];

    /// Get the number of items in the dataset.
    fn cardinality(&self) -> usize;

    /// Returns `true` if the dataset contains no items.
    fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    /// Consumes the dataset and returns its items as a `Vec` of `(Id, Item)` pairs.
    fn into_vec(self) -> Vec<(Self::Id, Self::Item)>;

    /// Creates a new dataset from a `Vec` of `(Id, Item)` pairs.
    fn from_vec(vec: Vec<(Self::Id, Self::Item)>) -> Self;

    /// Maps the dataset to a new dataset with potentially different Id and Item types.
    ///
    /// The given function can be used to transform each `(Id, Item)` pair into a new `(Id, Item)` pair.
    fn map<NewD, F>(self, f: F) -> NewD
    where
        NewD: Dataset,
        F: FnMut((Self::Id, Self::Item)) -> (NewD::Id, NewD::Item);
}

/// A simple implementation of `Dataset` for a `Vec` of `(Id, Item)` pairs.
impl<Id, Item> Dataset for Vec<(Id, Item)> {
    type Id = Id;
    type Item = Item;

    fn as_slice(&self) -> &[(Self::Id, Self::Item)] {
        self.as_slice()
    }

    fn as_mut_slice(&mut self) -> &mut [(Self::Id, Self::Item)] {
        self.as_mut_slice()
    }

    fn cardinality(&self) -> usize {
        self.len()
    }

    fn into_vec(self) -> Vec<(Self::Id, Self::Item)> {
        self
    }

    fn from_vec(vec: Vec<(Self::Id, Self::Item)>) -> Self {
        vec
    }

    fn map<NewD, F>(self, f: F) -> NewD
    where
        NewD: Dataset,
        F: FnMut((Self::Id, Self::Item)) -> (NewD::Id, NewD::Item),
    {
        NewD::from_vec(self.into_vec().into_iter().map(f).collect())
    }
}
