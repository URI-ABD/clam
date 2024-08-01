//! An adaptation of `Ball` that stores indices after reordering the dataset.

use core::fmt::Debug;

use distances::Number;
use rayon::prelude::*;

use crate::{
    adapter::{Adapter, ParAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::ParDataset,
    Ball, Cluster, Dataset, Permutable,
};

/// A variant of `Ball` that stores indices after reordering the dataset.
#[derive(Clone)]
pub struct OffBall<U: Number, S: Cluster<U>> {
    /// The `Cluster` type that the `OffsetBall` is based on.
    source: S,
    /// The children of the `Cluster`.
    children: Vec<(usize, U, Box<Self>)>,
    /// The parameters of the `Cluster`.
    params: Offset,
}

impl<U: Number, S: Cluster<U> + Debug> Debug for OffBall<U, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffsetBall")
            .field("source", &self.source)
            .field("children", &self.children.is_empty())
            .field("offset", &self.params.offset)
            .finish()
    }
}

impl<U: Number> OffBall<U, Ball<U>> {
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    pub fn from_ball_tree<I, D: Dataset<I, U> + Permutable>(ball: Ball<U>, data: &mut D) -> Self {
        let (root, indices) = Self::adapt(ball, None);
        data.permute(&indices);
        root
    }

    /// Parallel version of the `from_ball_tree` method.
    pub fn par_from_ball_tree<I, D: Dataset<I, U> + Permutable>(ball: Ball<U>, data: &mut D) -> Self {
        let (root, indices) = Self::par_adapt(ball, None);
        data.permute(&indices);
        root
    }

    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.params.offset
    }
}

impl<U: Number, S: Cluster<U>> Adapter<U, S, Offset> for OffBall<U, S> {
    fn adapt(source: S, params: Option<Offset>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (source, mut indices, children) = source.disassemble();
        let params = params.unwrap_or_default();

        let mut cluster = if children.is_empty() {
            Self::newly_adapted(source, Vec::new(), params)
        } else {
            let (arg_extrema, others) = children
                .into_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let (children, ret_indices) = params
                .child_params(&children)
                .into_iter()
                .zip(children)
                .map(|(p, c)| Self::adapt(*c, Some(p)))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let children = arg_extrema
                .into_iter()
                .zip(extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, Box::new(c)))
                .collect();

            indices = ret_indices.into_iter().flatten().collect();
            Self::newly_adapted(source, children, params)
        };

        // Update the indices of the important instances in the `Cluster`.
        cluster.set_arg_center(new_index(cluster.source.arg_center(), &indices, params.offset));
        cluster.set_arg_radial(new_index(cluster.source.arg_radial(), &indices, params.offset));
        for (p, _, _) in cluster.children_mut() {
            *p = new_index(*p, &indices, params.offset);
        }

        (cluster, indices)
    }

    fn newly_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: Offset) -> Self {
        Self {
            source,
            children,
            params,
        }
    }

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }
}

/// Helper for computing a new index after permutation of data.
fn new_index(i: usize, indices: &[usize], offset: usize) -> usize {
    offset
        + indices
            .iter()
            .position(|x| *x == i)
            .unwrap_or_else(|| unreachable!("This is a private function and we always pass a valid item."))
}

impl<U: Number, S: ParCluster<U>> ParAdapter<U, S, Offset> for OffBall<U, S> {
    fn par_adapt(source: S, params: Option<Offset>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (source, mut indices, children) = source.disassemble();
        let params = params.unwrap_or_default();

        let mut cluster = if children.is_empty() {
            Self {
                source,
                children: Vec::new(),
                params,
            }
        } else {
            let (arg_extrema, others) = children
                .into_iter()
                .map(|(a, b, c)| (a, (b, c)))
                .unzip::<_, _, Vec<_>, Vec<_>>();
            let (extents, children) = others.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();
            let (children, ret_indices) = params
                .child_params(&children)
                .into_par_iter()
                .zip(children.into_par_iter())
                .map(|(p, c)| Self::par_adapt(*c, Some(p)))
                .unzip::<_, _, Vec<_>, Vec<_>>();

            let children = arg_extrema
                .into_iter()
                .zip(extents)
                .zip(children)
                .map(|((a, b), c)| (a, b, Box::new(c)))
                .collect();

            indices = ret_indices.into_iter().flatten().collect();
            Self::newly_adapted(source, children, params)
        };

        // Update the indices of the important instances in the `Cluster`.
        cluster.set_arg_center(new_index(cluster.source.arg_center(), &indices, params.offset));
        cluster.set_arg_radial(new_index(cluster.source.arg_radial(), &indices, params.offset));
        for (p, _, _) in cluster.children_mut() {
            *p = new_index(*p, &indices, params.offset);
        }

        (cluster, indices)
    }
}

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone)]
struct Offset {
    /// The offset of the slice of indices of the `Cluster` in the reordered
    /// dataset.
    offset: usize,
}

impl<U: Number, S: Cluster<U>> Params<U, S> for Offset {
    fn child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self> {
        let mut offset = self.offset;
        child_balls
            .iter()
            .map(|child| {
                let params = Self { offset };
                offset += child.as_ref().cardinality();
                params
            })
            .collect()
    }
}

impl<U: Number, S: ParCluster<U>> ParParams<U, S> for Offset {
    fn par_child_params<B: AsRef<S>>(&self, child_balls: &[B]) -> Vec<Self> {
        // Since we need to keep track of the offset, we cannot parallelize this.
        self.child_params(child_balls)
    }
}

impl<U: Number, S: Cluster<U>> Cluster<U> for OffBall<U, S> {
    fn new<I, D: crate::Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
    where
        Self: Sized,
    {
        let (source, arg_radial) = S::new(data, indices, depth, seed);
        // TODO: Consider whether to reset indices of the ball.
        let vertex = Self {
            source,
            children: Vec::new(),
            params: Offset::default(),
        };
        (vertex, arg_radial)
    }

    fn disassemble(self) -> (Self, Vec<usize>, Vec<(usize, U, Box<Self>)>) {
        let indices = self.indices().collect();
        let Self {
            source,
            children,
            params,
        } = self;
        (
            Self {
                source,
                children: Vec::new(),
                params,
            },
            indices,
            children,
        )
    }

    fn depth(&self) -> usize {
        self.source.depth()
    }

    fn cardinality(&self) -> usize {
        self.source.cardinality()
    }

    fn arg_center(&self) -> usize {
        self.source.arg_center()
    }

    fn set_arg_center(&mut self, arg_center: usize) {
        self.source.set_arg_center(arg_center);
    }

    fn radius(&self) -> U {
        self.source.radius()
    }

    fn arg_radial(&self) -> usize {
        self.source.arg_radial()
    }

    fn set_arg_radial(&mut self, arg_radial: usize) {
        self.source.set_arg_radial(arg_radial);
    }

    fn lfd(&self) -> f32 {
        self.source.lfd()
    }

    fn indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.params.offset..(self.params.offset + self.cardinality())
    }

    fn set_indices(&mut self, indices: Vec<usize>) {
        let offset = indices[0];
        self.params.offset = offset;
    }

    fn children(&self) -> &[(usize, U, Box<Self>)] {
        self.children.as_slice()
    }

    fn children_mut(&mut self) -> &mut [(usize, U, Box<Self>)] {
        self.children.as_mut_slice()
    }

    fn set_children(mut self, children: Vec<(usize, U, Self)>) -> Self {
        self.children = children.into_iter().map(|(a, b, c)| (a, b, Box::new(c))).collect();
        self
    }

    fn find_extrema<I, D: crate::Dataset<I, U>>(&self, data: &D) -> Vec<usize> {
        self.source.find_extrema(data)
    }
}

impl<U: Number, S: Cluster<U>> PartialEq for OffBall<U, S> {
    fn eq(&self, other: &Self) -> bool {
        self.params.offset == other.params.offset && self.cardinality() == other.cardinality()
    }
}

impl<U: Number, S: Cluster<U>> Eq for OffBall<U, S> {}

impl<U: Number, S: Cluster<U>> PartialOrd for OffBall<U, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number, S: Cluster<U>> Ord for OffBall<U, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.params
            .offset
            .cmp(&other.params.offset)
            .then_with(|| other.cardinality().cmp(&self.cardinality()))
    }
}

impl<U: Number, S: Cluster<U>> std::hash::Hash for OffBall<U, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.params.offset, self.cardinality()).hash(state);
    }
}

impl<U: Number, S: ParCluster<U>> ParCluster<U> for OffBall<U, S> {
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize)
    where
        Self: Sized,
    {
        let (source, arg_radial) = S::par_new(data, indices, depth, seed);
        let vertex = Self {
            source,
            children: Vec::new(),
            params: Offset::default(),
        };
        (vertex, arg_radial)
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> Vec<usize> {
        self.source.par_find_extrema(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::{partition::ParPartition, FlatVec, Metric, Partition};

    use super::*;

    fn gen_tiny_data() -> Result<FlatVec<Vec<i32>, i32, usize>, String> {
        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        FlatVec::new_array(instances.clone(), metric)
    }

    fn check_permutation(root: &OffBall<i32, Ball<i32>>, data: &FlatVec<Vec<i32>, i32, usize>) -> bool {
        assert!(!root.children().is_empty());

        for cluster in root.subtree() {
            let radius = data.one_to_one(cluster.arg_center(), cluster.arg_radial());
            assert_eq!(cluster.radius(), radius);
        }

        true
    }

    #[test]
    fn permutation() -> Result<(), String> {
        let data = gen_tiny_data()?;

        let seed = Some(42);
        let criteria = |c: &Ball<i32>| c.depth() < 1;

        let root = Ball::new_tree(&data, &criteria, seed);
        let mut perm_data = data.clone();
        let root = OffBall::from_ball_tree(root, &mut perm_data);
        assert!(check_permutation(&root, &perm_data));

        let root = Ball::par_new_tree(&data, &criteria, seed);
        let mut perm_data = data.clone();
        let root = OffBall::par_from_ball_tree(root, &mut perm_data);
        assert!(check_permutation(&root, &perm_data));

        Ok(())
    }
}
