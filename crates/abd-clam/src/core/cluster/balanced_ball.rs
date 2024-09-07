//! A `Cluster` that provides a balanced clustering.

use distances::Number;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{dataset::ParDataset, Dataset};

use super::{partition::ParPartition, Ball, Cluster, ParCluster, Partition};

/// A `Cluster` that provides a balanced clustering.
#[derive(Clone, Serialize, Deserialize)]
pub struct BalancedBall<I, U: Number, D: Dataset<I, U>> {
    /// The inner `Ball` of the `BalancedBall`.
    pub(crate) ball: Ball<I, U, D>,
    /// The children of the `BalancedBall`.
    pub(crate) children: Vec<(usize, U, Box<Self>)>,
}

impl<I, U: Number, D: Dataset<I, U>> core::fmt::Debug for BalancedBall<I, U, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Ball")
            .field("depth", &self.ball.depth())
            .field("cardinality", &self.ball.cardinality())
            .field("radius", &self.ball.radius())
            .field("lfd", &self.ball.lfd())
            .field("arg_center", &self.ball.arg_center())
            .field("arg_radial", &self.ball.arg_radial())
            .field("indices", &self.ball.indices)
            .field("children", &self.children.is_empty())
            .finish()
    }
}

impl<I, U: Number, D: Dataset<I, U>> PartialEq for BalancedBall<I, U, D> {
    fn eq(&self, other: &Self) -> bool {
        self.ball.eq(&other.ball)
    }
}

impl<I, U: Number, D: Dataset<I, U>> Eq for BalancedBall<I, U, D> {}

impl<I, U: Number, D: Dataset<I, U>> PartialOrd for BalancedBall<I, U, D> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, U: Number, D: Dataset<I, U>> Ord for BalancedBall<I, U, D> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.ball.cmp(&other.ball)
    }
}

impl<I, U: Number, D: Dataset<I, U>> std::hash::Hash for BalancedBall<I, U, D> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.ball.hash(state);
    }
}

impl<I, U: Number, D: Dataset<I, U>> Cluster<I, U, D> for BalancedBall<I, U, D> {
    fn depth(&self) -> usize {
        self.ball.depth()
    }

    fn cardinality(&self) -> usize {
        self.ball.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.ball.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.ball.set_arg_center(arg_center);
    }

    fn radius(&self) -> U {
        self.ball.radius()
    }

    fn arg_radial(&self) -> usize {
        self.ball.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.ball.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.ball.lfd()
    }

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.ball.indices()
    }

    fn set_indices(&mut self, indices: Vec<usize>) {
        self.ball.set_indices(indices);
    }

    fn children(&self) -> &[(usize, U, Box<Self>)] {
        &self.children
    }

    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)] {
        &mut self.children
    }

    fn set_children(&mut self, children: Vec<(usize, U, Box<Self>)>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<(usize, U, Box<Self>)> {
        core::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.ball.distances_to_query(data, query)
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        self.ball.is_descendant_of(&other.ball)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParCluster<I, U, D> for BalancedBall<I, U, D> {
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        self.ball.par_distances_to_query(data, query)
    }
}

impl<I, U: Number, D: Dataset<I, U>> Partition<I, U, D> for BalancedBall<I, U, D> {
    fn new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self {
        let ball = Ball::new(data, indices, depth, seed);
        let children = Vec::new();
        Self { ball, children }
    }

    fn find_extrema(&self, data: &D) -> Vec<usize> {
        self.ball.find_extrema(data)
    }

    fn split_by_extrema(&self, data: &D, extrema: &[usize]) -> (Vec<Vec<usize>>, Vec<U>) {
        let mut instances = self.indices().filter(|&i| !extrema.contains(&i)).collect::<Vec<_>>();

        // Calculate the number of instances per child for a balanced split
        let num_per_child = instances.len() / extrema.len();
        let last_child_size = if instances.len() % extrema.len() == 0 {
            num_per_child
        } else {
            num_per_child + 1
        };
        let child_sizes =
            core::iter::once(last_child_size).chain(core::iter::repeat(num_per_child).take(extrema.len() - 1));

        // Initialize the child stacks with the extrema
        let mut child_stacks = extrema.iter().map(|&e| vec![(e, U::ZERO)]).collect::<Vec<_>>();
        for (child_stack, s) in child_stacks.iter_mut().zip(child_sizes) {
            // Calculate the distances to the instances from the extremum
            let mut distances = Dataset::one_to_many(data, child_stack[0].0, &instances);
            distances.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Less));

            // Remove the closest instances from the distances and add them to the child stack
            child_stack.extend(distances.split_off(instances.len() - s));

            // Update the instances for the next child
            instances = distances.into_iter().map(|(i, _)| i).collect();
        }

        // Unzip the child stacks into the indices and calculate the extent of each child
        child_stacks
            .into_iter()
            .map(|stack| {
                let (indices, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let extent = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (indices, extent)
            })
            .unzip()
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>> ParPartition<I, U, D> for BalancedBall<I, U, D> {
    fn par_new(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> Self {
        let ball = Ball::par_new(data, indices, depth, seed);
        let children = Vec::new();
        Self { ball, children }
    }

    fn par_find_extrema(&self, data: &D) -> Vec<usize> {
        self.ball.par_find_extrema(data)
    }

    fn par_split_by_extrema(&self, data: &D, extrema: &[usize]) -> (Vec<Vec<usize>>, Vec<U>) {
        let mut instances = self.indices().filter(|&i| !extrema.contains(&i)).collect::<Vec<_>>();

        // Calculate the number of instances per child for a balanced split
        let num_per_child = instances.len() / extrema.len();
        let last_child_size = if instances.len() % extrema.len() == 0 {
            num_per_child
        } else {
            num_per_child + 1
        };
        let child_sizes =
            core::iter::once(last_child_size).chain(core::iter::repeat(num_per_child).take(extrema.len() - 1));

        // Initialize the child stacks with the extrema
        let mut child_stacks = extrema.iter().map(|&e| vec![(e, U::ZERO)]).collect::<Vec<_>>();
        for (child_stack, s) in child_stacks.iter_mut().zip(child_sizes) {
            // Calculate the distances to the instances from the extremum
            let mut distances = ParDataset::par_one_to_many(data, child_stack[0].0, &instances);
            distances.par_sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Less));

            // Remove the closest instances from the distances and add them to the child stack
            child_stack.extend(distances.split_off(instances.len() - s));

            // Update the instances for the next child
            instances = distances.into_iter().map(|(i, _)| i).collect();
        }

        // Unzip the child stacks into the indices and calculate the extent of each child
        child_stacks
            .into_par_iter()
            .map(|stack| {
                let (indices, distances) = stack.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
                let extent = distances
                    .into_iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Less))
                    .unwrap_or_else(|| unreachable!("Cannot find the maximum distance"));
                (indices, extent)
            })
            .unzip()
    }
}

#[cfg(test)]
mod tests {
    use crate::{partition::ParPartition, Cluster, Dataset, FlatVec, Metric, Partition};

    use super::BalancedBall;

    type F = FlatVec<Vec<i32>, i32, usize>;
    type B = BalancedBall<Vec<i32>, i32, F>;

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
        let root = B::new(&data, &indices, 0, seed);
        let arg_r = root.arg_radial();

        assert_eq!(arg_r, data.cardinality() - 1);
        assert_eq!(root.depth(), 0);
        assert_eq!(root.cardinality(), 5);
        assert_eq!(root.arg_center(), 2);
        assert_eq!(root.radius(), 12);
        assert_eq!(root.arg_radial(), arg_r);
        assert!(root.children().is_empty());
        assert_eq!(root.indices().collect::<Vec<_>>(), indices);

        let root = B::par_new(&data, &indices, 0, seed);
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
        assert_eq!(indices, &[0, 2, 1, 4, 3]);

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

        let root = B::new_tree(&data, &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality(), "{root:?}");
        assert!(check_partition(&root));

        let root = B::par_new_tree(&data, &criteria, seed);
        assert_eq!(root.indices().count(), data.cardinality(), "{root:?}");
        assert!(check_partition(&root));

        Ok(())
    }
}
