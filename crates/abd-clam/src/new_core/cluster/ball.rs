//! The most basic representation of a `Cluster` is a metric-`Ball`.

use core::fmt::Debug;

use distances::Number;

use crate::{
    new_core::{Dataset, ParDataset},
    utils,
};

use super::{children::Children, Cluster, IndexStore, ParPartition, Partition, LFD};

/// A metric-`Ball` is a collection of instances that are within a certain
/// distance of a center.
pub struct Ball<U: Number> {
    /// Parameters used for creating the `Ball`.
    depth: usize,
    /// The number of instances in the `Ball`.
    cardinality: usize,
    /// The radius of the `Ball`.
    radius: U,
    /// The local fractal dimension of the `Ball`.
    lfd: f32,
    /// The index of the center instance.
    arg_center: usize,
    /// The index of the instance that is the furthest from the center.
    arg_radial: usize,
    /// The indices of the instances in the `Ball`.
    index_store: IndexStore,
    /// The children of the `Ball`.
    children: Option<Children<U, Self>>,
}

impl<U: Number> Debug for Ball<U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ball")
            .field("depth", &self.depth)
            .field("cardinality", &self.cardinality)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("arg_center", &self.arg_center)
            .field("arg_radial", &self.arg_radial)
            .field("indices", &self.index_store)
            .field("children", &self.children.is_some())
            .finish()
    }
}

impl<U: Number> PartialEq for Ball<U> {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.index_store == other.index_store && self.cardinality == other.cardinality
    }
}

impl<U: Number> Eq for Ball<U> {}

impl<U: Number> PartialOrd for Ball<U> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for Ball<U> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.depth
            .cmp(&other.depth)
            .then_with(|| self.index_store.cmp(&other.index_store))
            .then_with(|| self.cardinality.cmp(&other.cardinality))
    }
}

impl<U: Number> Cluster<U> for Ball<U> {
    fn new<I, D: Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
    where
        Self: Sized,
    {
        if indices.is_empty() {
            unreachable!("Cannot create a Ball with no instances")
        }

        let cardinality = indices.len();

        let samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (cardinality.as_f64().sqrt()) as usize;
            Dataset::choose_unique(data, indices, n, seed)
        };

        let arg_center = Dataset::median(data, &samples);

        let distances = Dataset::one_to_many(data, arg_center, indices);
        let &(arg_radial, radius) = distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let distances = distances.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
        let lfd_scale = radius.half();
        let lfd = LFD::from_radial_distances(&distances, lfd_scale);

        let c = Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            index_store: IndexStore::EveryCluster(indices.to_vec()),
            children: None,
        };

        (c, arg_radial)
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn cardinality(&self) -> usize {
        self.cardinality
    }

    fn arg_center(&self) -> usize {
        self.arg_center
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.arg_center = arg_center;
    }

    fn radius(&self) -> U {
        self.radius
    }

    fn arg_radial(&self) -> usize {
        self.arg_radial
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.arg_radial = arg_radial;
    }

    fn lfd(&self) -> f32 {
        self.lfd
    }

    fn index_store(&self) -> &IndexStore {
        &self.index_store
    }

    fn set_index_store(&mut self, indices: IndexStore) {
        self.index_store = indices;
    }

    fn children(&self) -> Option<&Children<U, Self>>
    where
        Self: Sized,
    {
        self.children.as_ref()
    }

    fn children_mut(&mut self) -> Option<&mut Children<U, Self>>
    where
        Self: Sized,
    {
        self.children.as_mut()
    }
}

impl<U: Number> Partition<U> for Ball<U> {
    fn find_extrema<I, D: Dataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>) {
        let indices = self.index_store().indices(self);
        let l_distances = Dataset::one_to_many(data, self.arg_radial, &indices);

        let &(arg_l, d) = l_distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let extrema = vec![arg_l, self.arg_radial];
        let remaining_instances = indices
            .into_iter()
            .filter(|&i| i != self.arg_radial && i != arg_l)
            .collect();
        let distances = vec![vec![U::ZERO, d], vec![d, U::ZERO]];

        (extrema, remaining_instances, distances)
    }

    fn set_children(mut self, children: Children<U, Self>) -> Self
    where
        Self: Sized,
    {
        self.children = Some(children);
        self
    }
}

impl<U: Number> ParPartition<U> for Ball<U> {
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize)
    where
        Self: Sized,
    {
        if indices.is_empty() {
            unreachable!("Cannot create a Ball with no instances")
        }

        let cardinality = indices.len();

        let samples = if cardinality < 100 {
            indices.to_vec()
        } else {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let n = (cardinality.as_f64().sqrt()) as usize;
            ParDataset::par_choose_unique(data, indices, n, seed)
        };

        let arg_center = ParDataset::par_median(data, &samples);

        let distances = ParDataset::par_one_to_many(data, arg_center, indices);
        let &(arg_radial, radius) = distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let distances = distances.into_iter().map(|(_, d)| d).collect::<Vec<_>>();
        let lfd = utils::compute_lfd(radius, &distances);

        let c = Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            index_store: IndexStore::EveryCluster(indices.to_vec()),
            children: None,
        };

        (c, arg_radial)
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>) {
        let indices = self.index_store().indices(self);
        let l_distances = ParDataset::par_one_to_many(data, self.arg_radial, &indices);

        let &(arg_l, d) = l_distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        let extrema = vec![arg_l, self.arg_radial];
        let remaining_instances = indices
            .into_iter()
            .filter(|&i| i != self.arg_radial && i != arg_l)
            .collect();
        let distances = vec![vec![U::ZERO, d], vec![d, U::ZERO]];

        (extrema, remaining_instances, distances)
    }
}

#[cfg(test)]
mod tests {
    use crate::new_core::{FlatVec, Metric, Permutable};

    use super::*;

    fn gen_tiny_data() -> Result<FlatVec<Vec<i32>, i32, usize>, String> {
        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        FlatVec::new_array(instances.clone(), metric)
    }

    #[test]
    fn new() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let seed = Some(42);
        let (root, arg_r) = Ball::new(&data, &indices, 0, seed);

        assert_eq!(arg_r, data.cardinality() - 1);
        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality(), 5);
        assert_eq!(root.arg_center(), 2);
        assert_eq!(root.radius(), 12);
        assert_eq!(root.arg_radial(), arg_r);
        assert!(root.children().is_none());
        assert_eq!(root.index_store().indices(&root), indices);

        let (root, arg_r) = Ball::par_new(&data, &indices, 0, seed);

        assert_eq!(arg_r, data.cardinality() - 1);
        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality(), 5);
        assert_eq!(root.arg_center(), 2);
        assert_eq!(root.radius(), 12);
        assert_eq!(root.arg_radial(), arg_r);
        assert!(root.children().is_none());
        assert_eq!(root.index_store().indices(&root), indices);

        Ok(())
    }

    fn check_partition(root: &Ball<i32>, indices: &[usize]) -> bool {
        assert!(root.children().is_some());
        assert_eq!(indices, &[0, 1, 2, 4, 3]);

        let children = root.children().unwrap().clusters();
        assert_eq!(children.len(), 2);
        for &c in &children {
            assert_eq!(c.depth(), 1);
            assert!(c.children().is_none());
        }

        let (left, right) = (children[0], children[1]);

        assert_eq!(left.cardinality(), 3);
        assert_eq!(left.arg_center(), 1);
        assert_eq!(left.radius(), 4);
        assert!([0, 2].contains(&left.arg_radial()));

        assert_eq!(right.cardinality(), 2);
        assert_eq!(right.radius(), 8);
        assert!([3, 4].contains(&right.arg_center()));
        assert!([3, 4].contains(&right.arg_radial()));

        true
    }

    #[test]
    fn partition() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let seed = Some(42);
        let criteria = |c: &Ball<i32>| c.depth() < 1;

        let (root, indices) = Ball::new(&data, &indices, 0, seed)
            .0
            .partition(&data, indices, &criteria, seed);
        assert_eq!(indices.len(), data.cardinality());
        assert!(check_partition(&root, &indices));

        let (root, indices) = Ball::par_new(&data, &indices, 0, seed)
            .0
            .par_partition(&data, indices, &criteria, seed);
        assert_eq!(indices.len(), data.cardinality());
        assert!(check_partition(&root, &indices));

        Ok(())
    }

    #[test]
    fn tree() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let seed = Some(42);
        let criteria = |c: &Ball<i32>| c.depth() < 1;

        let (root, indices) = Ball::new_tree(&data, &criteria, seed);
        assert_eq!(indices.len(), data.cardinality());
        assert!(check_partition(&root, &indices));

        let (root, indices) = Ball::par_new_tree(&data, &criteria, seed);
        assert_eq!(indices.len(), data.cardinality());
        assert!(check_partition(&root, &indices));

        Ok(())
    }

    fn check_permutation(root: &Ball<i32>, indices: &[usize]) -> bool {
        assert!(root.children().is_some());
        assert_eq!(indices, &[0, 1, 2, 4, 3]);

        let children = root.children().unwrap().clusters();
        assert_eq!(children.len(), 2);
        for &c in &children {
            assert_eq!(c.depth(), 1);
            assert!(c.children().is_none());
        }

        let (left, right) = (children[0], children[1]);

        assert_eq!(left.cardinality(), 3);
        assert_eq!(left.arg_center(), 1);
        assert_eq!(left.radius(), 4);
        assert!([0, 2].contains(&left.arg_radial()));
        assert_eq!(left.indices_post_permutation(), 0..3);

        assert_eq!(right.cardinality(), 2);
        assert_eq!(right.radius(), 8);
        assert!([3, 4].contains(&right.arg_center()));
        assert!([3, 4].contains(&right.arg_radial()));
        assert_eq!(right.indices_post_permutation(), 3..5);

        true
    }

    #[test]
    fn permutation() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let seed = Some(42);
        let criteria = |c: &Ball<i32>| c.depth() < 1;

        let mut perm_data = data.clone();
        let (root, indices) = Ball::new_tree_and_permute(&mut perm_data, &criteria, seed);
        assert_eq!(indices.len(), perm_data.cardinality());
        assert_eq!(indices, perm_data.permutation());
        assert!(check_permutation(&root, &indices));

        let mut perm_data = data.clone();
        let (root, indices) = Ball::par_new_tree_and_permute(&mut perm_data, &criteria, seed);
        assert_eq!(indices.len(), perm_data.cardinality());
        assert_eq!(indices, perm_data.permutation());
        assert!(check_permutation(&root, &indices));

        Ok(())
    }
}
