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
#[derive(Clone, Serialize, Deserialize)]
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
    pub(crate) indices: Vec<usize>,
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
            .field("children", &!self.children.is_empty())
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

    fn set_children(&mut self, children: Vec<(usize, U, Box<Self>)>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<(usize, U, Box<Self>)> {
        core::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.query_to_many(query, &self.indices)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        let indices = other.indices().collect::<std::collections::HashSet<_>>();
        self.indices().all(|i| indices.contains(&i))
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParCluster<I, U, D> for Ball<I, U, D> {
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.par_query_to_many(query, &self.indices().collect::<Vec<_>>())
    }
}

impl<I, U: Number, D: Dataset<I, U>> Partition<I, U, D> for Ball<I, U, D> {
    fn new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self
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
            let n = if cardinality < 10_100 {
                // We use the square root of the cardinality as the number of samples
                (cardinality - 100).as_f64().sqrt().as_u64() as usize
            } else {
                // We use the logarithm of the cardinality as the number of samples
                #[allow(clippy::cast_possible_truncation)]
                let n = (cardinality - 10_100).as_f64().log2().as_u64() as usize;
                n + 100
            };

            let n = n + 100;
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

        Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            indices: indices.to_vec(),
            children: Vec::new(),
            _id: PhantomData,
        }
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
    fn par_new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self
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
            let n = if cardinality < 10_100 {
                // We use the square root of the cardinality as the number of samples
                (cardinality - 100).as_f64().sqrt().as_u64() as usize
            } else {
                // We use the logarithm of the cardinality as the number of samples
                #[allow(clippy::cast_possible_truncation)]
                let n = (cardinality - 10_100).as_f64().log2().as_u64() as usize;
                n + 100
            };

            let n = n + 100;
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

        Self {
            depth,
            cardinality,
            radius,
            lfd,
            arg_center,
            arg_radial,
            indices: indices.to_vec(),
            children: Vec::new(),
            _id: PhantomData,
        }
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
    use std::collections::HashSet;

    use distances::number::{Addition, Multiplication};

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

    fn gen_pathological_line() -> FlatVec<f64, f64, usize> {
        let min_delta = 1e-12;
        let mut delta = min_delta;
        let mut line = vec![0_f64];

        while line.len() < 900 {
            let last = *line.last().unwrap();
            line.push(last + delta);
            delta *= 2.0;
            delta += min_delta;
        }

        let distance_fn = |x: &f64, y: &f64| x.abs_diff(*y);
        let metric = Metric::new(distance_fn, false);
        FlatVec::new(line, metric).unwrap()
    }

    #[test]
    fn new() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let indices = (0..data.cardinality()).collect::<Vec<_>>();
        let seed = Some(42);
        let root = Ball::new(&data, &indices, 0, seed);
        let arg_r = root.arg_radial();

        assert_eq!(arg_r, data.cardinality() - 1);
        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality(), 5);
        assert_eq!(root.arg_center(), 2);
        assert_eq!(root.radius(), 12);
        assert_eq!(root.arg_radial(), arg_r);
        assert!(root.children().is_empty());
        assert_eq!(root.indices().collect::<Vec<_>>(), indices);

        let root = Ball::par_new(&data, &indices, 0, seed);
        let arg_r = root.arg_radial();

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

    #[test]
    fn partition_further() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let seed = Some(42);
        let criteria_one = |c: &B| c.depth() < 1;
        let criteria_two = |c: &B| c.depth() < 2;

        let mut root = Ball::new_tree(&data, &criteria_one, seed);
        for leaf in root.leaves() {
            assert_eq!(leaf.depth(), 1);
        }
        root.partition_further(&data, &criteria_two, seed);
        for leaf in root.leaves() {
            assert_eq!(leaf.depth(), 2);
        }

        let mut root = Ball::par_new_tree(&data, &criteria_one, seed);
        for leaf in root.leaves() {
            assert_eq!(leaf.depth(), 1);
        }
        root.par_partition_further(&data, &criteria_two, seed);
        for leaf in root.leaves() {
            assert_eq!(leaf.depth(), 2);
        }

        Ok(())
    }

    #[test]
    fn tree_iterative() {
        let data = gen_pathological_line();

        let seed = Some(42);
        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;

        let mut intermediate_depth = crate::MAX_RECURSION_DEPTH;
        let intermediate_criteria = |c: &Ball<_, _, _>| c.depth() < intermediate_depth && criteria(c);
        let mut root = Ball::new_tree(&data, &intermediate_criteria, seed);

        while root.leaves().into_iter().any(|l| !l.is_singleton()) {
            intermediate_depth += crate::MAX_RECURSION_DEPTH;
            let intermediate_criteria = |c: &Ball<_, _, _>| c.depth() < intermediate_depth && criteria(c);
            root.partition_further(&data, &intermediate_criteria, seed);
        }

        assert!(!root.is_leaf());
    }

    #[test]
    fn trim_and_graft() -> Result<(), String> {
        let line = (0..1024).collect();
        let distance_fn = |x: &u32, y: &u32| x.abs_diff(*y);
        let metric = Metric::new(distance_fn, false);
        let data = FlatVec::new(line, metric)?;

        let seed = Some(42);
        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let root = Ball::new_tree(&data, &criteria, seed);

        let target_depth = 4;
        let mut grafted_root = root.clone();
        let children = grafted_root.trim_at_depth(target_depth);

        let leaves = grafted_root.leaves();
        assert_eq!(leaves.len(), 2.powi(target_depth as i32));
        assert_eq!(leaves.len(), children.len());

        grafted_root.graft_at_depth(target_depth, children);
        assert_eq!(grafted_root, root);
        for (l, c) in root.subtree().into_iter().zip(grafted_root.subtree()) {
            assert_eq!(l, c);
        }

        Ok(())
    }

    #[test]
    fn numbered_subtree() {
        let data = (0..1024).collect::<Vec<_>>();
        let distance_fn = |x: &u32, y: &u32| x.abs_diff(*y);
        let metric = Metric::new(distance_fn, false);
        let data = FlatVec::new(data, metric).unwrap();

        let seed = Some(42);
        let criteria = |c: &Ball<_, _, _>| c.cardinality() > 1;
        let root = Ball::new_tree(&data, &criteria, seed);

        let mut numbered_subtree = root.clone().take_subtree();
        let indices = numbered_subtree.iter().map(|(_, i, _, _)| *i).collect::<HashSet<_>>();
        assert_eq!(indices.len(), numbered_subtree.len());

        for i in 0..numbered_subtree.len() {
            assert!(indices.contains(&i));
        }

        numbered_subtree.sort_by(|(_, i, _, _), (_, j, _, _)| i.cmp(j));
        let numbered_subtree = numbered_subtree
            .into_iter()
            .map(|(a, _, _, b)| (a, b))
            .collect::<Vec<_>>();

        for (ball, child_indices) in &numbered_subtree {
            for &(i, _) in child_indices {
                let (child, _) = &numbered_subtree[i];
                assert!(child.is_descendant_of(ball), "{ball:?} is not parent of {child:?}");
            }
        }

        // let root_list = root.clone().as_indexed_list();
        // let re_root = Ball::from_indexed_list(root_list);
        // assert_eq!(root, re_root);

        // for (l, r) in root.subtree().into_iter().zip(re_root.subtree()) {
        //     assert_eq!(l, r);
        // }
    }
}
