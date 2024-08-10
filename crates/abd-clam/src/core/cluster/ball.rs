//! The most basic representation of a `Cluster` is a metric-`Ball`.

use core::fmt::Debug;

use std::{hash::Hash, marker::PhantomData};

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{dataset::ParDataset, utils, Dataset};

use super::{
    partition::{ParPartition, Partition},
    Cluster, ParCluster, LFD,
};

/// A metric-`Ball` is a collection of instances that are within a certain
/// distance of a center.
#[derive(Clone, Deserialize, Serialize)]
pub struct Ball<I, U: Number, D: Dataset<I, U>> {
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
    indices: Vec<usize>,
    /// The children of the `Ball`.
    children: Vec<(usize, U, Box<Self>)>,
    /// Phantom data to satisfy the compiler.
    _id: PhantomData<(I, D)>,
}

impl<I, U: Number, D: Dataset<I, U>> Debug for Ball<I, U, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ball")
            .field("depth", &self.depth)
            .field("cardinality", &self.cardinality)
            .field("radius", &self.radius)
            .field("lfd", &self.lfd)
            .field("arg_center", &self.arg_center)
            .field("arg_radial", &self.arg_radial)
            .field("indices", &self.indices)
            .field("children", &self.children.is_empty())
            .finish()
    }
}

impl<I, U: Number, D: Dataset<I, U>> PartialEq for Ball<I, U, D> {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.cardinality == other.cardinality && self.indices == other.indices
    }
}

impl<I, U: Number, D: Dataset<I, U>> Eq for Ball<I, U, D> {}

impl<I, U: Number, D: Dataset<I, U>> PartialOrd for Ball<I, U, D> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, U: Number, D: Dataset<I, U>> Ord for Ball<I, U, D> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.depth
            .cmp(&other.depth)
            .then_with(|| self.cardinality.cmp(&other.cardinality))
            .then_with(|| self.indices.cmp(&other.indices))
    }
}

impl<I, U: Number, D: Dataset<I, U>> Hash for Ball<I, U, D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // We hash the `indices` field
        self.indices.hash(state);
    }
}

impl<I, U: Number, D: Dataset<I, U>> Cluster<I, U, D> for Ball<I, U, D> {
    fn disassemble(mut self) -> (Self, Vec<usize>, Vec<(usize, U, Box<Self>)>) {
        let indices = self.indices;
        self.indices = Vec::new();
        let children = self.children;
        self.children = Vec::new();
        (self, indices, children)
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

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.indices.iter().copied()
    }

    fn set_indices(&mut self, indices: Vec<usize>) {
        self.indices = indices;
    }

    fn children(&self) -> &[(usize, U, Box<Self>)] {
        self.children.as_slice()
    }

    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)] {
        self.children.as_mut_slice()
    }

    fn set_children(mut self, children: Vec<(usize, U, Self)>) -> Self {
        self.children = children.into_iter().map(|(i, d, c)| (i, d, Box::new(c))).collect();
        self
    }

    fn distances(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.query_to_many(query, &self.indices)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParCluster<I, U, D> for Ball<I, U, D> {
    fn par_distances(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.par_query_to_many(query, &self.indices().collect::<Vec<_>>())
    }
}

impl<I, U: Number, D: Dataset<I, U>> Partition<I, U, D> for Ball<I, U, D> {
    fn new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
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
            #[allow(clippy::cast_possible_truncation)]
            let n = cardinality.as_f64().sqrt().as_u64() as usize;
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
            indices: indices.to_vec(),
            children: Vec::new(),
            _id: PhantomData,
        };

        (c, arg_radial)
    }

    fn find_extrema(&self, data: &D) -> Vec<usize> {
        let l_distances = Dataset::one_to_many(data, self.arg_radial, &self.indices);

        let &(arg_l, _) = l_distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        vec![arg_l, self.arg_radial]
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParPartition<I, U, D> for Ball<I, U, D> {
    fn par_new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
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
            #[allow(clippy::cast_possible_truncation)]
            let n = cardinality.as_f64().sqrt().as_u64() as usize;
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
            indices: indices.to_vec(),
            children: Vec::new(),
            _id: PhantomData,
        };

        (c, arg_radial)
    }

    fn par_find_extrema(&self, data: &D) -> Vec<usize> {
        let l_distances = ParDataset::par_one_to_many(data, self.arg_radial, &self.indices);

        let &(arg_l, _) = l_distances
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
            .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));

        vec![arg_l, self.arg_radial]
    }
}

#[cfg(test)]
mod tests {
    use crate::{partition::ParPartition, Cluster, Dataset, FlatVec, Metric, Partition};

    use super::Ball;

    type F = FlatVec<Vec<i32>, i32, usize>;
    type B = Ball<Vec<i32>, i32, F>;

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
        assert!(root.children().is_empty());
        assert_eq!(root.indices().collect::<Vec<_>>(), indices);

        let (root, arg_r) = Ball::par_new(&data, &indices, 0, seed);

        assert_eq!(arg_r, data.cardinality() - 1);
        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality(), 5);
        assert_eq!(root.arg_center(), 2);
        assert_eq!(root.radius(), 12);
        assert_eq!(root.arg_radial(), arg_r);
        assert!(root.children().is_empty());
        assert_eq!(root.indices().collect::<Vec<_>>(), indices);

        Ok(())
    }

    fn check_partition(root: &B) -> bool {
        let indices = root.indices().collect::<Vec<_>>();

        assert!(!root.children().is_empty());
        assert_eq!(indices, &[0, 1, 2, 4, 3]);

        let children = root.child_clusters().collect::<Vec<_>>();
        assert_eq!(children.len(), 2);
        for &c in &children {
            assert_eq!(c.depth(), 1);
            assert!(c.children().is_empty());
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
        let criteria = |c: &B| c.depth() < 1;

        let root = Ball::new(&data, &indices, 0, seed)
            .0
            .partition(&data, indices.clone(), &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality());
        assert!(check_partition(&root));

        let root = Ball::par_new(&data, &indices, 0, seed)
            .0
            .par_partition(&data, indices, &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality());
        assert!(check_partition(&root));

        Ok(())
    }

    #[test]
    fn tree() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let seed = Some(42);
        let criteria = |c: &B| c.depth() < 1;

        let root = Ball::new_tree(&data, &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality());
        assert!(check_partition(&root));

        let root = Ball::par_new_tree(&data, &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality());
        assert!(check_partition(&root));

        Ok(())
    }
}
