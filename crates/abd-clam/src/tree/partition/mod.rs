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
    /// - If the number of `items` is greater than 2, this function will find the geometric median of the items (using an approximate method for larger numbers
    ///   of items) and use it as the center of the cluster. It will swap the center item to the 0th index in the `items` slice. It will then compute the radius
    ///   and LFD of the cluster.
    ///
    /// # Arguments
    ///
    /// - `depth` - The depth of the cluster in the tree.
    /// - `center_index` - The index of the center item in the full list of `items` in the tree. At build time, this is the offset of the start of the local
    ///   slice of `items` for the cluster from the start of the full list of `items` in the tree. This method will find the center item and swap it to the 0th
    ///   index in the local slice of `items`, so that the `center_index` will become correct after the cluster is created.
    /// - `parent_center_index` - The index of the parent cluster's center item, or `None` if this is the root cluster.
    /// - `items` - The local slice of items that this cluster represents. This slice is mutable because the center item will be swapped to the front and, if
    ///   the cluster is partitioned further, the remaining items will be reordered in place and split into contiguous slices for the child clusters.
    /// - `metric` - The distance function to use.
    /// - `annotator` - A function that takes an unannotated, `Cluster<T, ()>`, and returns an annotation of type `A` for that cluster. This is called before
    ///   deciding whether to partition the cluster further, so the annotation can be used in the decision.
    /// - `should_partition` - A function that decides whether to partition this cluster further.
    /// - `strategy` - The partitioning strategy to use. See the [`PartitionStrategy`] docs for details on the available strategies and their parameters.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    ///   - The newly constructed `Cluster`.
    ///   - A vector of tuples, one for each child cluster, each tuple containing:
    ///     - The center index of the child cluster in the full list of `items` in the tree.
    ///     - The slice of items belonging to the child cluster.
    ///
    /// # WARNING
    ///
    /// This function should never be made public because it assumes that `items` is non-empty. This is checked *once* when creating the `Tree` and is,
    /// thereafter, ensured by the logic of the partitioning algorithms.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new<'a, Id, I, M, Ann, P>(
        depth: usize,
        center_index: usize,
        parent_center_index: Option<usize>,
        items: &'a mut [(Id, I)],
        metric: &M,
        annotator: &Ann,
        should_partition: &P,
        strategy: &PartitionStrategy,
    ) -> (Self, Splits<'a, Id, I>)
    where
        T: DistanceValue,
        M: Fn(&I, &I) -> T,
        Ann: Fn(&Cluster<T, ()>) -> A,
        P: Fn(&Self) -> bool,
    {
        profi::prof!("Cluster::new");
        ftlog::debug!("Creating a new cluster with cardinality {}", items.len());

        // Create a `Cluster` with some dummy values, which will be updated as needed.
        let mut cluster = Cluster {
            depth,
            center_index,
            cardinality: items.len(),
            radius: T::zero(), // Will be updated after finding the radius if there are enough items.
            lfd: 1.0,          // Will be updated after finding the radius and LFD if there are enough items.
            children: None,    // Will be updated if the cluster is partitioned further.
            annotation: (),    // The annotation is computed after finding the radius and LFD but before deciding whether to partition further.
            parent_center_index,
        };

        if cluster.cardinality == 1 {
            // For a singleton cluster, the radius is 0 and LFD is 1 by definition.
            let cluster = cluster.change_annotation_with(annotator).0;
            return (cluster, Vec::new());
        } else if cluster.cardinality == 2 {
            // For a cluster with two items, the radius is the distance between the two items and LFD is 1 by definition.
            cluster.radius = metric(&items[0].1, &items[1].1);
            let cluster = cluster.change_annotation_with(annotator).0;
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

        // Update the cluster's radius, LFD and annotation.
        cluster.radius = radius;
        cluster.lfd = lfd_estimate(&radial_distances, radius);
        let mut cluster = cluster.change_annotation_with(annotator).0;

        // Check if we should partition this cluster further. If not, return the cluster with no splits.
        if !should_partition(&cluster) {
            return (cluster, Vec::new());
        }

        // Split the `items` slice into contiguous sub-slices for child clusters.
        let (span, mut splits) = strategy.split(&mut items[1..], metric, radius_index);
        for (ci, _) in &mut splits {
            // Increment the center indices of the child clusters by the center index to update it from being relative to the local slice of `items` for the
            // cluster to being relative to the full list of `items` in the tree.
            *ci += center_index;
        }

        // Set the center indices of the child clusters and the span of this cluster. The child clusters will be created later in `Tree::new`.
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
    profi::prof!("swap_center_to_front");
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        ftlog::debug!("Finding the geometric median among {} items", items.len());
        let center_index = geometric_median(items, metric);
        ftlog::debug!("The geometric median is at index {center_index} in the local slice");
        items.swap(0, center_index);
    }
}
