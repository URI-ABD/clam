//! Partitioning algorithms for the tree and clusters.

use crate::{
    Cluster, DistanceValue,
    utils::{geometric_median, lfd_estimate},
};

mod par_partition;
pub mod strategy;

use strategy::{PartitionStrategy, Splits};

impl<T, A> Cluster<T, A> {
    /// Creates a new `Cluster` and, if it should be partitioned, applies the partitioning strategy to reorder and split the items for the child clusters.
    ///
    /// - If the number of `items` is 1, that item is the center, the radius is 0, and the LFD is 1.
    /// - If the number of `items` is 2, the 0th item is the center, the radius is the distance between the two items, and the LFD is 1.
    /// - If the number of `items` is greater than 2, this function will find the geometric median of the items (using an approximate method for large number of
    ///   items) and use it as the center of the cluster. It will swap the center item to the 0th index in the `items` slice. It will then compute the radius of
    ///   the cluster as the maximum distance from the center to any other item, and compute the LFD of the cluster.
    ///
    /// # Arguments
    ///
    /// - `items` - The local slice of items belonging to the cluster.
    /// - `metric` - The distance function to use.
    /// - `strategy` - The partitioning strategy to use.
    ///
    /// # Side Effects
    ///
    /// - The center item, i.e. the geometric median, will be moved to the 0th index in the `items` slice.
    /// - If the `strategy` decides to partition the cluster, the remaining `items` will be reordered in place and split into contiguous slices for the child
    ///   clusters.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///   - The new `Cluster`.
    ///   - A vector of tuples, one for each child cluster, each tuple containing:
    ///     - The center index of the child cluster in the full list of `items` in the tree.
    ///     - The slice of items belonging to the child cluster.
    ///
    /// # WARNING
    ///
    /// This function should never be made public because it:
    ///
    /// - assumes that `items` is non-empty. This is checked *once* when creating the `Tree` and ensured by the logic of the partitioning algorithms.
    /// - initializes the `annotation` to a dummy value. The cluster is annotated later in `Tree::new`.
    /// - sets the `depth`, `center_index` and `parent_center_index` to 0. These are updated later in `Tree::new`.
    /// - sets the `child_center_indices` with respect to the local slice of `items` for the cluster. These are updated with an offset to be with respect to the
    ///   full list of `items` in the tree later in `Tree::new`.
    pub(crate) fn new<'a, Id, I, M, P>(items: &'a mut [(Id, I)], metric: &M, strategy: &PartitionStrategy<P>) -> (Self, Splits<'a, Id, I>)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
        P: Fn(&Self) -> bool,
    {
        ftlog::debug!("Creating a new cluster with cardinality {}", items.len());

        // Create a `Cluster` with some dummy values, which will be updated as needed.
        let mut cluster = Self {
            depth: 0, // Will be updated in `Tree::new`.
            center_index: 0, // Will be updated after finding the geometric median if there are enough items, and again in `Tree::new`.
            cardinality: items.len(),
            radius: T::zero(), // Will be updated after finding the radius if there are enough items.
            lfd: 1.0, // Will be updated after finding the radius and LFD if there are enough items.
            children: None, // Will be updated if the `strategy` decides to partition this cluster further.
            #[expect(unsafe_code)]
            // SAFETY: This is a private function and the annotation is later set in `Tree::new` before being used.
            annotation: unsafe { core::mem::zeroed() },
            parent_center_index: None, // Will be updated in `Tree::new`.
        };

        if cluster.cardinality == 1 {
            // For a singleton cluster, the radius is 0 and LFD is 1 by definition, so no changes are needed to the dummy values.
            return (cluster, Vec::new());
        } else if cluster.cardinality == 2 {
            // For a cluster with two items, the radius is the distance between the two items and LFD is 1 by definition.
            cluster.radius = metric(&items[0].1, &items[1].1);
            return (cluster, Vec::new());
        }

        // Find and move the center (geometric median) to the 0th index in the local slice of `items`.
        let n = num_items_for_geometric_median(cluster.cardinality);
        swap_center_to_front(&mut items[..n], metric);

        // Compute the radius and the index of the item that defines the radius (the item farthest from the center).
        let radial_distances = items.iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();
        let (radius_index, radius) = radial_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has enough elements"), |(i, &d)| (i, d));

        // Update the cluster's radius and LFD.
        cluster.radius = radius;
        cluster.lfd = lfd_estimate(&radial_distances, radius);

        // Check if we should partition this cluster further based on the provided strategy. If not, return the cluster with no splits.
        if !strategy.should_partition(&cluster) {
            return (cluster, Vec::new());
        }

        // Split the `items` slice into contiguous sub-slices for child clusters.
        let (span, splits) = strategy.split(&mut items[1..], metric, radius_index);

        // The `child_center_indices` are the indices of the centers of the child clusters relative to the original `items` slice. These will need to be updated
        // with an offset to be with respect to the full list of `items` in the tree later in `Tree::new`.
        let child_center_indices = splits.iter().map(|&(c_index, _)| c_index).collect::<Vec<_>>();
        cluster.children = Some((child_center_indices.into_boxed_slice(), span));

        (cluster, splits)
    }
}

/// Computes the number of items to use for finding the geometric median.
#[expect(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
pub fn num_items_for_geometric_median(cluster_cardinality: usize) -> usize {
    if cluster_cardinality <= 100 {
        ftlog::debug!("Using all {cluster_cardinality} items for finding the exact geometric median.");
        cluster_cardinality
    } else {
        let n = if cluster_cardinality <= 10_100 {
            let base = 100;
            let sqrt = ((cluster_cardinality - 100) as f64).sqrt();
            base + sqrt as usize
        } else {
            let base = 200;
            let log = ((cluster_cardinality - 10_100) as f64).log2();
            base + log as usize
        };
        ftlog::debug!("Using a random sample of size {n} out of {cluster_cardinality} items for finding an approximate geometric median.");
        n
    }
}

/// Moves the center item (geometric median) to the 0th index in the slice.
pub fn swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
where
    T: DistanceValue,
    M: Fn(&I, &I) -> T,
{
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        ftlog::debug!("Finding the geometric median among {} items", items.len());
        let center_index = geometric_median(items, metric);
        ftlog::debug!("The geometric median is at index {center_index} in the local slice");
        items.swap(0, center_index);
    }
}
