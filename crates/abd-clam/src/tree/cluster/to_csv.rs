//! Exporting the `Cluster` and its information to various formats.

use crate::DistanceValue;

use super::Cluster;

/// The number of features to include in the CSV export.
const NUM_CLUSTER_FEATURES: usize = 8;

/// These methods, gated behind the `serde` feature, allow exporting the `Cluster` and its subtree to a CSV file.
impl<T, A> Cluster<T, A>
where
    T: DistanceValue,
{
    /// Returns a CSV header string for the cluster information.
    #[must_use]
    pub const fn csv_header() -> [&'static str; NUM_CLUSTER_FEATURES] {
        [
            "depth",
            "center_index",
            "cardinality",
            "radius",
            "lfd",
            "num_children",
            "span",
            "parent_center_index",
        ]
    }

    /// Returns a row of CSV data representing the cluster's information.
    pub fn csv_row(&self) -> [String; NUM_CLUSTER_FEATURES] {
        [
            self.depth.to_string(),
            self.center_index.to_string(),
            self.cardinality().to_string(),
            self.radius().to_string(),
            self.lfd().to_string(),
            self.child_center_indices().map_or(0, <[_]>::len).to_string(),
            self.span().map_or_else(|| T::zero().to_string(), |s| s.to_string()),
            self.parent_center_index().unwrap_or(0).to_string(),
        ]
    }
}
