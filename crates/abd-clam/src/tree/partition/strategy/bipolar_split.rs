//! Splitting a slice of items into two partitions.

use rayon::prelude::*;

use crate::DistanceValue;

/// Two partitions of items created by bipolar split.
#[derive(Debug)]
pub struct BipolarSplit<'a, Id, I, T> {
    /// The left partition of items. The 0th item is the left pole.
    pub l_items: &'a mut [(Id, I)],
    /// The right partition of items. The 0th item is the right pole.
    pub r_items: &'a mut [(Id, I)],
    /// The span of the partition (distance between the two poles).
    pub span: T,
    /// Distances from the left pole to the items in the left partition (excluding the left pole itself).
    pub l_distances: Vec<T>,
    /// Distances from the right pole to the items in the right partition (excluding the right pole itself).
    pub r_distances: Vec<T>,
}

/// The information we have about an initial pole for bipolar partitioning.
#[derive(Clone, Debug)]
pub enum InitialPole<T> {
    /// We have the index of the item farthest from the center to use as the left pole.
    RadialIndex(usize),
    /// The pole is the first item in the slice, and we have precomputed distances from it to all other items.
    Distances(Vec<T>),
}

impl<'a, Id, I, T> BipolarSplit<'a, Id, I, T>
where
    T: DistanceValue,
{
    /// Splits the given items into two partitions based on their distances to two poles.
    ///
    /// The two poles are chosen as follows:
    ///
    /// - If the `initial_pole` is `RadialIndex`, the item at that index is the left pole.
    /// - If the `initial_pole` is `Distances`, the 0th item is the left pole, and the provided distances are used as distances from it to all other items.
    /// - The right pole is then chosen as the item farthest from the left pole.
    ///
    /// The `span` of the partition is defined as the distance between the two poles.
    ///
    /// # Returns
    ///
    /// The `BipolarSplit` containing the two partitions of items, their distances to their respective poles, and the span of the partition.
    pub fn new<M>(items: &'a mut [(Id, I)], metric: &M, initial_pole: InitialPole<T>) -> Self
    where
        M: Fn(&I, &I) -> T,
    {
        if items.len() == 2 {
            ftlog::debug!("Splitting a slice with only two items");
            // If there are only two items, just return them as the two partitions.
            let span = metric(&items[0].1, &items[1].1);
            let (l_items, r_items) = items.split_at_mut(1);
            let (l_distances, r_distances) = (vec![span], vec![span]);
            return Self {
                l_items,
                r_items,
                span,
                l_distances,
                r_distances,
            };
        }
        ftlog::debug!("Splitting a slice with {} items", items.len());

        // Determine the left pole and swap it to the 0th index if necessary.
        let left_pole_index = match initial_pole {
            InitialPole::RadialIndex(i) => i,
            InitialPole::Distances(distances) => {
                // The left pole is the farthest item from the center among the distances provided. Find its index and move it to the 0th index in the slice.
                distances
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, d)| crate::utils::MaxItem((), *d))
                    .map_or(0, |(i, _)| i)
            }
        };
        ftlog::debug!("Left pole index: {left_pole_index}");
        // Move the left pole to the 0th index in the slice
        items.swap(0, left_pole_index);

        // Compute distances from the left pole to all other items
        let mut left_distances = items.iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();

        // Find the item farthest from the left pole
        let (right_pole_index, span) = left_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has at least two elements"), |(i, &d)| (i + 1, d));
        ftlog::debug!("Right pole index: {right_pole_index}, Span: {span}");

        // Move the right pole and its distance to the ends of their respective slices
        let last = items.len() - 1;
        items.swap(right_pole_index, last);
        left_distances.swap(right_pole_index - 1, last - 1);

        // Compute the distance from the right pole to all items
        let right_pole = &items[items.len() - 1].1;
        let mut left_right_distances = items
            .iter()
            .skip(1) // Skip the left pole
            .zip(left_distances)
            .take(items.len() - 2) // Exclude the right pole
            .map(|((_, item), l)| (l, metric(right_pole, item)))
            .collect::<Vec<_>>();

        // Reorder the items in place by their distances to the two poles
        let mid = reorder_items_in_place(&mut items[1..last], &mut left_right_distances) + 1; // +1 to account for the left pole at index 0
        ftlog::debug!("Mid index after reordering: {mid}");

        // Split the items and distances into the left and right partitions
        let (l_items, r_items) = items.split_at_mut(mid);
        let (l_distances, r_distances) = left_right_distances.split_at(mid - 1); // -1 to account for the left pole at index 0
        let l_distances = l_distances.iter().map(|&(l, _)| l).collect::<Vec<_>>();
        let r_distances = {
            // The first distance is just a placeholder for the right pole itself. We will swap it to the front to match with the right pole's position.
            let mut r_distances = core::iter::once(T::zero()).chain(r_distances.iter().map(|&(_, r)| r)).collect::<Vec<_>>();
            r_distances.swap_remove(0); // Remove the placeholder and move the first actual distance to the front
            r_distances
        };

        // Move the right pole to the 0th index of the right partition
        r_items.swap(0, r_items.len() - 1);

        ftlog::debug!("Left partition size: {}, Right partition size: {}", l_items.len(), r_items.len());

        Self {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        }
    }
}

impl<'a, Id, I, T> BipolarSplit<'a, Id, I, T>
where
    Id: Send + Sync,
    I: Send + Sync,
    T: DistanceValue + Send + Sync,
{
    /// Parallel version of [`Self::new`].
    pub fn par_new<M>(items: &'a mut [(Id, I)], metric: &M, initial_pole: InitialPole<T>) -> Self
    where
        M: Fn(&I, &I) -> T + Send + Sync,
    {
        if items.len() == 2 {
            ftlog::debug!("Splitting a slice with only two items");
            // If there are only two items, just return them as the two partitions.
            let span = metric(&items[0].1, &items[1].1);
            let (l_items, r_items) = items.split_at_mut(1);
            let (l_distances, r_distances) = (vec![span], vec![span]);
            return Self {
                l_items,
                r_items,
                span,
                l_distances,
                r_distances,
            };
        }
        ftlog::debug!("Splitting a slice with {} items", items.len());

        // Determine the left pole and swap it to the 0th index if necessary.
        let left_pole_index = match initial_pole {
            InitialPole::RadialIndex(i) => i,
            InitialPole::Distances(distances) => {
                // The left pole is the farthest item from the center among the distances provided. Find its index and move it to the 0th index in the slice.
                distances
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, d)| crate::utils::MaxItem((), *d))
                    .map_or(0, |(i, _)| i)
            }
        };
        ftlog::debug!("Left pole index: {left_pole_index}");
        // Move the left pole to the 0th index in the slice
        items.swap(0, left_pole_index);

        // Compute distances from the left pole to all other items
        let mut left_distances = items.par_iter().skip(1).map(|(_, item)| metric(&items[0].1, item)).collect::<Vec<_>>();

        // Find the item farthest from the left pole
        let (right_pole_index, span) = left_distances
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| crate::utils::MaxItem(i, d))
            .map_or_else(|| unreachable!("items has at least two elements"), |(i, &d)| (i + 1, d));
        ftlog::debug!("Right pole index: {right_pole_index}, Span: {span}");

        // Move the right pole and its distance to the left pole to the end of their respective slices
        let last = items.len() - 1;
        items.swap(right_pole_index, last);
        left_distances.swap(right_pole_index - 1, last - 1);

        // Compute the distance from the right pole to all items
        let right_pole = &items[items.len() - 1].1;
        let mut left_right_distances = items
            .par_iter()
            .skip(1) // Skip the left pole
            .zip(left_distances)
            .take(items.len() - 2) // Exclude the right pole
            .map(|((_, item), l)| (l, metric(right_pole, item)))
            .collect::<Vec<_>>();

        // Reorder the items in place by their distances to the two poles
        let mid = reorder_items_in_place(&mut items[1..last], &mut left_right_distances) + 1; // +1 to account for the left pole at index 0
        ftlog::debug!("Mid index after reordering: {mid}");

        // Split the items and distances into the left and right partitions
        let (l_items, r_items) = items.split_at_mut(mid);
        let (l_distances, r_distances) = left_right_distances.split_at(mid - 1); // -1 to account for the left pole at index 0
        let l_distances = l_distances.iter().map(|&(l, _)| l).collect::<Vec<_>>();
        let r_distances = {
            // The first distance is just a placeholder for the right pole itself. We will swap it to the front to match with the right pole's position.
            let mut r_distances = core::iter::once(T::zero()).chain(r_distances.iter().map(|&(_, r)| r)).collect::<Vec<_>>();
            r_distances.swap_remove(0); // Remove the placeholder and move the first actual distance to the front
            r_distances
        };

        // Move the right pole to the 0th index of the right partition
        r_items.swap(0, r_items.len() - 1);

        ftlog::debug!("Left partition size: {}, Right partition size: {}", l_items.len(), r_items.len());

        Self {
            l_items,
            r_items,
            span,
            l_distances,
            r_distances,
        }
    }
}

/// Reorder the slice of items so that the items closer to left pole are on the left side of the slice and items closer to the right pole are on the right side,
/// returning `mid`, the index of the first item for the right pole.
///
/// # WARNING
///
/// This assumes that `items` and `distances` have the same length.
pub fn reorder_items_in_place<Id, I, T>(items: &mut [(Id, I)], distances: &mut [(T, T)]) -> usize
where
    T: DistanceValue,
{
    ftlog::debug!("Reordering {} items in place", items.len());

    let mut left = 0;
    let mut right = distances.len() - 1;

    // TODO(Najib): After testing, use unsafe code to remove bounds checks while indexing
    while left < right {
        // Increment `left` until we find an item for the right pole
        while left < distances.len() && distances[left].0 <= distances[left].1 {
            left += 1;
        }

        // Decrement `right` until we find an item for the left pole
        while right > 0 && distances[right].0 > distances[right].1 {
            right -= 1;
        }

        // If the two indices have crossed, we are done
        if left >= right {
            break;
        }

        // swap the items at the two indices
        items.swap(left, right);
        distances.swap(left, right);
        left += 1;
        right -= 1;
    }

    // TODO(Najib): Check if this last loop is even necessary

    // Increment `left` until we find the first item for the right pole
    while left < distances.len() && distances[left].0 <= distances[left].1 {
        left += 1;
    }

    left
}

/// Compute the triangle projections of two polar distances `l` and `r` onto the line connecting the two poles, returning the distance from the left pole to the
/// projection of the item on the line connecting the two poles.
#[expect(unused)]
fn triangle_projection<T>(l: T, r: T, span: T) -> f64
where
    T: DistanceValue,
{
    // Convert the distances to f64 for easier computation of the triangle projection.
    let l = l.to_f64().unwrap_or_else(|| unreachable!("Distance values should be convertible to f64"));
    let r = r.to_f64().unwrap_or_else(|| unreachable!("Distance values should be convertible to f64"));
    let span = span.to_f64().unwrap_or_else(|| unreachable!("Distance values should be convertible to f64"));

    // Use the law of cosines to compute the angle between the line segment connecting the poles and the line segment connecting the left pole to the item.
    let cos_theta = r.mul_add(-r, l.mul_add(l, span * span)) / (2.0 * l * span);

    // Use the distance from the left pole to the item as the hypotenuse of a right triangle, and the angle we just computed to find the base of that triangle.
    // Then, normalize the projection by the span to get a value between 0 and 1.
    (l * cos_theta) / span
}
