//! A `Cluster` in a `Tree` for use in CLAM.

mod getters;
mod setters;
#[cfg(feature = "serde")]
mod to_csv;

/// A `Cluster` is a node in the `Tree` that represents a subset of the items in the `Tree`.
///
/// It contains information about the subset including:
///
/// - `depth`: The depth of the cluster in the tree, with the root at depth 0.
/// - `center_index`: The index of the center item in the `items` array of the `Tree`.
/// - `cardinality`: The number of items in the subtree rooted at this cluster, including the center item.
/// - `radius`: The distance from the center item to the furthest item in the cluster.
/// - `lfd`: The Local Fractal Dimension of the cluster.
/// - `children`: If the cluster was partitioned, this contains the indices of child centers and the span of the cluster (the distance between the two poles
///   used to partition the cluster).
/// - `annotation`: Arbitrary data associated with this cluster.
/// - `parent_center_index`: The index of the center of the parent cluster. For the root cluster, this will be None.
///
/// Each `Cluster` is uniquely identified by its `center_index` and `Cluster`s can be ordered based on this index.
///
/// # Generics
///
/// - `T`: The type of the distance values between items.
/// - `A`: The type of the annotation data associated with this cluster.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", expect(clippy::unsafe_derive_deserialize))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize, databuf::Encode, databuf::Decode))]
#[must_use]
pub struct Cluster<T, A> {
    /// Depth of this cluster in the tree, with root at depth 0.
    pub(crate) depth: usize,
    /// Index of the center item in the `items` array of the `Tree`.
    pub(crate) center_index: usize,
    /// Number of items in the subtree rooted at this cluster, including the center item.
    pub(crate) cardinality: usize,
    /// The distance from the center item to the furthest item in the cluster.
    pub(crate) radius: T,
    /// The Local Fractal Dimension of the cluster.
    pub(crate) lfd: f64,
    /// The indices of child centers and the span of this cluster, if it was partitioned. The span is the distance between the two poles used to partition the cluster.
    pub(crate) children: Option<(Box<[usize]>, T)>,
    /// Arbitrary data associated with this cluster.
    pub(crate) annotation: A,
    /// The index of the center of the parent cluster. For the root cluster, this will be None.
    pub(crate) parent_center_index: Option<usize>,
}

impl<T, A> PartialEq for Cluster<T, A> {
    fn eq(&self, other: &Self) -> bool {
        self.center_index == other.center_index
    }
}

impl<T, A> Eq for Cluster<T, A> {}

impl<T, A> PartialOrd for Cluster<T, A> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T, A> Ord for Cluster<T, A> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.center_index.cmp(&other.center_index)
    }
}
