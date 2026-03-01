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
    pub(crate) fn par_new<'a, Id, I, M, Ann, P>(
        depth: usize,
        center_index: usize,
        parent_center_index: Option<usize>,
        items: &'a mut [(Id, I)],
        metric: &M,
        annotator: &Ann,
        strategy: &PartitionStrategy<P>,
    ) -> (Self, Splits<'a, Id, I>)
    where
        T: DistanceValue + Send + Sync,
        A: Send + Sync,
        Id: Send + Sync,
        I: Send + Sync,
        M: Fn(&I, &I) -> T + Send + Sync,
        Ann: Fn(&Self) -> A + Send + Sync,
        P: Fn(&Self) -> bool + Send + Sync,
    {
        ftlog::debug!("Creating a new cluster with cardinality {}", items.len());

        // Create a `Cluster` with some dummy values, which will be updated as needed.
        let mut cluster = Self {
            depth,
            center_index,
            cardinality: items.len(),
            radius: T::zero(), // Will be updated after finding the radius if there are enough items.
            lfd: 1.0, // Will be updated after finding the radius and LFD if there are enough items.
            children: None, // Will be updated if the `strategy` decides to partition this cluster further.
            #[expect(unsafe_code)]
            // SAFETY: The annotation is set just before deciding whether to partition the cluster further, so it is guaranteed to be set before potentially
            // being used in the partitioning strategy.
            annotation: unsafe { core::mem::zeroed() },
            parent_center_index,
        };

        if cluster.cardinality == 1 {
            // For a singleton cluster, the radius is 0 and LFD is 1 by definition.
            cluster.annotation = annotator(&cluster);
            return (cluster, Vec::new());
        } else if cluster.cardinality == 2 {
            // For a cluster with two items, the radius is the distance between the two items and LFD is 1 by definition.
            cluster.radius = metric(&items[0].1, &items[1].1);
            cluster.annotation = annotator(&cluster);
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
        cluster.annotation = annotator(&cluster);

        // Check if we should partition this cluster further based on the provided strategy. If not, return the cluster with no splits.
        if !strategy.should_partition(&cluster) {
            return (cluster, Vec::new());
        }

        // Split the `items` slice into contiguous sub-slices for child clusters.
        let (span, mut splits) = strategy.par_split(&mut items[1..], metric, radius_index);
        for (ci, _) in &mut splits {
            // Increment the center indices of the child clusters by the center index to update it from being relative to the local slice of `items` for the
            // cluster to being relative to the full list of `items` in the tree.
            *ci += center_index;
        }

        // The `child_center_indices` are the indices of the centers of the child clusters relative to the original `items` slice. These will need to be updated
        // with an offset to be with respect to the full list of `items` in the tree later in `Tree::new`.
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
    if items.len() > 2 {
        // Find the index of the item with the minimum total distance to all other items.
        ftlog::debug!("Finding the geometric median among {} items", items.len());
        let center_index = par_geometric_median(items, metric);
        ftlog::debug!("The geometric median is at index {center_index} in the local slice");
        items.swap(0, center_index);
    }
}
