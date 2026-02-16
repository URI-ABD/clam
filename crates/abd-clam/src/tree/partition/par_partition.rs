//! Methods for recursively partitioning a `Cluster` to build a `Tree`.

use rayon::prelude::*;

use crate::{
    Cluster, DistanceValue,
    utils::{lfd_estimate, par_geometric_median},
};

use super::{
    num_items_for_geometric_median,
    strategy::{PartitionStrategy, Splits},
};

impl<T, A> Cluster<T, A> {
    /// Parallel version of [`Self::new`].
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn par_new<'a, Id, I, M, Ann, P>(
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
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        Ann: Fn(&Cluster<T, ()>) -> A + Send + Sync,
        P: Fn(&Self) -> bool + Send + Sync,
    {
        profi::prof!("Cluster::par_new");
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
        par_swap_center_to_front(&mut items[..n], metric);

        // Compute the radius and the index of the item that defines the radius (the item farthest from the center).
        let radial_distances = items.par_iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();
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
        let (span, mut splits) = strategy.par_split(&mut items[1..], metric, radius_index);
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

/// Moves the center item (geometric median) to the 0th index in the slice.
pub fn par_swap_center_to_front<Id, I, T, M>(items: &mut [(Id, I)], metric: &M)
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
    M: Fn(&I, &I) -> T + Send + Sync,
{
    profi::prof!("par_swap_center_to_front");
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        ftlog::debug!("Finding the geometric median among {} items", items.len());
        let center_index = par_geometric_median(items, metric);
        ftlog::debug!("The geometric median is at index {center_index} in the local slice");
        items.swap(0, center_index);
    }
}
