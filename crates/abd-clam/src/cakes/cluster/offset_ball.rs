//! An adaptation of `Ball` that stores indices after reordering the dataset.

use core::fmt::Debug;
use std::marker::PhantomData;

use distances::Number;
use serde::{Deserialize, Serialize};

use crate::{
    adapter::{Adapter, BallAdapter, ParAdapter, ParBallAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::ParDataset,
    Ball, Cluster, Dataset, Permutable,
};

/// A variant of `Ball` that stores indices after reordering the dataset.
#[derive(Clone, Serialize, Deserialize)]
pub struct OffBall<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> {
    /// The `Cluster` type that the `OffsetBall` is based on.
    source: S,
    /// The children of the `Cluster`.
    children: Vec<(usize, U, Box<Self>)>,
    /// The parameters of the `Cluster`.
    params: Offset,
    /// Phantom data to satisfy the compiler.
    _id: PhantomData<(I, D)>,
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D> + Debug> Debug for OffBall<I, U, D, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OffBall")
            .field("source", &self.source)
            .field("children", &!self.children.is_empty())
            .field("offset", &self.params.offset)
            .finish()
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> OffBall<I, U, D, S> {
    /// Returns the offset of the `Cluster`.
    #[must_use]
    pub const fn offset(&self) -> usize {
        self.params.offset
    }
}

impl<I, U: Number, D: Dataset<I, U> + Permutable> BallAdapter<I, U, D, D, Offset> for OffBall<I, U, D, Ball<I, U, D>> {
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn from_ball_tree(ball: Ball<I, U, D>, mut data: D, _: bool) -> (Self, D) {
        let mut root = Self::adapt_tree_iterative(ball, None);
        data.permute(&root.source.indices);
        root.source.clear_indices();
        (root, data)
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U> + Permutable> ParBallAdapter<I, U, D, D, Offset>
    for OffBall<I, U, D, Ball<I, U, D>>
{
    /// Creates a new `OffsetBall` tree from a `Ball` tree.
    fn par_from_ball_tree(ball: Ball<I, U, D>, mut data: D, _: bool) -> (Self, D) {
        let mut root = Self::par_adapt_tree_iterative(ball, None);
        data.permute(&root.source.indices);
        root.source.clear_indices();
        (root, data)
    }
}

impl<I, U: Number, D: Dataset<I, U> + Permutable, S: Cluster<I, U, D>> Adapter<I, U, D, D, S, Offset>
    for OffBall<I, U, D, S>
{
    fn new_adapted(source: S, children: Vec<(usize, U, Box<Self>)>, params: Offset) -> Self {
        Self {
            source,
            children,
            params,
            _id: PhantomData,
        }
    }

    fn post_traversal(&mut self) {
        // Update the indices of the important instances in the `Cluster`.
        let offset = self.params.offset;
        let indices = self.source.indices().collect::<Vec<_>>();
        self.set_arg_center(new_index(self.source.arg_center(), &indices, offset));
        self.set_arg_radial(new_index(self.source.arg_radial(), &indices, offset));
        for (p, _, _) in self.children_mut() {
            *p = new_index(*p, &indices, offset);
        }
    }

    fn source(&self) -> &S {
        &self.source
    }

    fn source_mut(&mut self) -> &mut S {
        &mut self.source
    }

    fn take_source(self) -> S {
        self.source
    }

    fn params(&self) -> &Offset {
        &self.params
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

impl<I: Send + Sync, U: Number, D: ParDataset<I, U> + Permutable, S: ParCluster<I, U, D>>
    ParAdapter<I, U, D, D, S, Offset> for OffBall<I, U, D, S>
{
    fn par_post_traversal(&mut self) {
        self.post_traversal();
    }
}

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub struct Offset {
    /// The offset of the slice of indices of the `Cluster` in the reordered
    /// dataset.
    offset: usize,
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Params<I, U, D, D, S> for Offset {
    fn child_params(&self, children: &[S]) -> Vec<Self> {
        let mut offset = self.offset;
        children
            .iter()
            .map(|child| {
                let params = Self { offset };
                offset += child.cardinality();
                params
            })
            .collect()
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParParams<I, U, D, D, S> for Offset {
    fn par_child_params(&self, children: &[S]) -> Vec<Self> {
        // Since we need to keep track of the offset, we cannot parallelize this.
        self.child_params(children)
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Cluster<I, U, D> for OffBall<I, U, D, S> {
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

    fn set_children(&mut self, children: Vec<(usize, U, Box<Self>)>) {
        self.children = children;
    }

    fn take_children(&mut self) -> Vec<(usize, U, Box<Self>)> {
        std::mem::take(&mut self.children)
    }

    fn distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.query_to_many(query, &self.indices().collect::<Vec<_>>())
    }

    fn is_descendant_of(&self, other: &Self) -> bool {
        let range = other.params.offset..(other.params.offset + other.cardinality());
        range.contains(&self.offset()) && self.cardinality() <= other.cardinality()
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialEq for OffBall<I, U, D, S> {
    fn eq(&self, other: &Self) -> bool {
        self.params.offset == other.params.offset && self.cardinality() == other.cardinality()
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Eq for OffBall<I, U, D, S> {}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> PartialOrd for OffBall<I, U, D, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> Ord for OffBall<I, U, D, S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.params
            .offset
            .cmp(&other.params.offset)
            .then_with(|| other.cardinality().cmp(&self.cardinality()))
    }
}

impl<I, U: Number, D: Dataset<I, U>, S: Cluster<I, U, D>> std::hash::Hash for OffBall<I, U, D, S> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.params.offset, self.cardinality()).hash(state);
    }
}

impl<I: Send + Sync, U: Number, D: ParDataset<I, U>, S: ParCluster<I, U, D>> ParCluster<I, U, D>
    for OffBall<I, U, D, S>
{
    fn par_distances_to_query(&self, data: &D, query: &I) -> Vec<(usize, U)> {
        data.par_query_to_many(query, &self.indices().collect::<Vec<_>>())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        adapter::{BallAdapter, ParBallAdapter},
        Ball, Cluster, Dataset, FlatVec, Metric, Partition,
    };

    use super::OffBall;

    type Fv = FlatVec<Vec<i32>, i32, usize>;
    type B = Ball<Vec<i32>, i32, Fv>;
    type Ob = OffBall<Vec<i32>, i32, Fv, B>;

    fn gen_tiny_data() -> Result<FlatVec<Vec<i32>, i32, usize>, String> {
        let instances = vec![vec![1, 2], vec![3, 4], vec![5, 6], vec![7, 8], vec![11, 12]];
        let distance_function = |a: &Vec<i32>, b: &Vec<i32>| distances::vectors::manhattan(a, b);
        let metric = Metric::new(distance_function, false);
        FlatVec::new_array(instances.clone(), metric)
    }

    fn check_permutation(root: &Ob, data: &FlatVec<Vec<i32>, i32, usize>) -> bool {
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
        let criteria = |c: &B| c.depth() < 1;

        let ball = Ball::new_tree(&data, &criteria, seed);

        let (root, perm_data) = OffBall::from_ball_tree(ball.clone(), data.clone(), false);
        assert!(check_permutation(&root, &perm_data));

        let (root, perm_data) = OffBall::par_from_ball_tree(ball, data, false);
        assert!(check_permutation(&root, &perm_data));

        Ok(())
    }
}
