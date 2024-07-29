//! An adaptation of `Ball` that stores indices after reordering the dataset.

use distances::Number;

use crate::{
    adapter::{Adapter, ParAdapter, ParParams, Params},
    cluster::ParCluster,
    dataset::ParDataset,
    Ball, Children, Cluster, Dataset, Permutable,
};

/// A variant of `Ball` that stores indices after reordering the dataset.
#[derive(Debug)]
pub struct OffsetBall<U: Number> {
    /// The `Ball` of the `Cluster`.
    ball: Ball<U>,
    /// The children of the `Cluster`.
    children: Option<Children<U, Self>>,
    /// The parameters of the `Cluster`.
    params: OffsetParams,
}

impl<U: Number> OffsetBall<U> {
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

impl<U: Number> Adapter<U, OffsetParams> for OffsetBall<U> {
    fn adapt(ball: Ball<U>, params: Option<OffsetParams>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (ball, mut indices, children) = ball.deconstruct();
        let params = params.unwrap_or_default();

        let mut cluster = if let Some(children) = children {
            let (mut children, ret_indices) = children.adapt(&params);
            children
                .arg_extrema_mut()
                .iter_mut()
                .for_each(|p| *p = params.offset + position_of(p, &ret_indices));
            let cluster = Self {
                ball,
                children: Some(children),
                params,
            };
            indices = ret_indices;
            cluster
        } else {
            Self {
                ball,
                children: None,
                params,
            }
        };

        let arg_center = position_of(&cluster.ball.arg_center(), &indices);
        cluster.set_arg_center(params.offset + arg_center);

        let arg_radial = position_of(&cluster.ball.arg_radial(), &indices);
        cluster.set_arg_radial(params.offset + arg_radial);

        (cluster, indices)
    }

    fn ball(&self) -> &Ball<U> {
        &self.ball
    }

    fn ball_mut(&mut self) -> &mut Ball<U> {
        &mut self.ball
    }
}

/// Returns the position of an item in a slice.
fn position_of<T: Eq>(item: &T, slice: &[T]) -> usize {
    slice
        .iter()
        .position(|x| x == item)
        .unwrap_or_else(|| unreachable!("This is a private function and we always pass a valid item."))
}

impl<U: Number> ParAdapter<U, OffsetParams> for OffsetBall<U> {
    fn par_adapt(ball: Ball<U>, params: Option<OffsetParams>) -> (Self, Vec<usize>)
    where
        Self: Sized,
    {
        let (ball, mut indices, children) = ball.deconstruct();
        let params = params.unwrap_or_default();

        let mut cluster = if let Some(children) = children {
            let (mut children, ret_indices) = children.par_adapt(&params);
            children
                .arg_extrema_mut()
                .iter_mut()
                .for_each(|p| *p = params.offset + position_of(p, &ret_indices));
            let cluster = Self {
                ball,
                children: Some(children),
                params,
            };
            indices = ret_indices;
            cluster
        } else {
            Self {
                ball,
                children: None,
                params,
            }
        };

        let arg_center = position_of(&cluster.ball.arg_center(), &indices);
        cluster.set_arg_center(params.offset + arg_center);

        let arg_radial = position_of(&cluster.ball.arg_radial(), &indices);
        cluster.set_arg_radial(params.offset + arg_radial);

        (cluster, indices)
    }
}

/// Parameters for the `OffsetBall`.
#[derive(Debug, Default, Copy, Clone)]
pub struct OffsetParams {
    /// The offset of the slice of indices of the `Cluster` in the reordered
    /// dataset.
    offset: usize,
}

impl<U: Number> Params<U> for OffsetParams {
    fn child_params<B: AsRef<Ball<U>>>(&self, child_balls: &[B]) -> Vec<Self> {
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

impl<U: Number> ParParams<U> for OffsetParams {
    fn par_child_params<B: AsRef<Ball<U>>>(&self, child_balls: &[B]) -> Vec<Self> {
        // Since we need to keep track of the offset, we cannot parallelize this.
        self.child_params(child_balls)
    }
}

impl<U: Number> Cluster<U> for OffsetBall<U> {
    fn new<I, D: crate::Dataset<I, U>>(data: &D, indices: &[usize], depth: usize, seed: Option<u64>) -> (Self, usize)
    where
        Self: Sized,
    {
        let (ball, arg_radial) = Ball::new(data, indices, depth, seed);
        // TODO: Consider whether to reset indices of the ball.
        let vertex = Self {
            ball,
            children: None,
            params: OffsetParams::default(),
        };
        (vertex, arg_radial)
    }

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
        self.params.offset..(self.params.offset + self.cardinality())
    }

    fn set_indices(&mut self, _: Vec<usize>) {
        todo!()
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

    fn set_children(mut self, children: Children<U, Self>) -> Self
    where
        Self: Sized,
    {
        self.children = Some(children);
        self
    }

    fn find_extrema<I, D: crate::Dataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>) {
        self.ball.find_extrema(data)
    }
}

impl<U: Number> PartialEq for OffsetBall<U> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: Should this be offset and cardinality?
        self.ball == other.ball
    }
}

impl<U: Number> Eq for OffsetBall<U> {}

impl<U: Number> PartialOrd for OffsetBall<U> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<U: Number> Ord for OffsetBall<U> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // TODO: This should definitely be offset and cardinality.
        self.ball.cmp(&other.ball)
    }
}

impl<U: Number> std::hash::Hash for OffsetBall<U> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.params.offset, self.cardinality()).hash(state);
    }
}

impl<U: Number> ParCluster<U> for OffsetBall<U> {
    fn par_new<I: Send + Sync, D: ParDataset<I, U>>(
        data: &D,
        indices: &[usize],
        depth: usize,
        seed: Option<u64>,
    ) -> (Self, usize)
    where
        Self: Sized,
    {
        let (ball, arg_radial) = Ball::par_new(data, indices, depth, seed);
        let vertex = Self {
            ball,
            children: None,
            params: OffsetParams::default(),
        };
        (vertex, arg_radial)
    }

    fn par_find_extrema<I: Send + Sync, D: ParDataset<I, U>>(&self, data: &D) -> (Vec<usize>, Vec<usize>, Vec<Vec<U>>) {
        self.ball.par_find_extrema(data)
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

    fn check_permutation(root: &OffsetBall<i32>, data: &FlatVec<Vec<i32>, i32, usize>) -> bool {
        assert!(root.children().is_some());

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
        let root = OffsetBall::from_ball_tree(root, &mut perm_data);
        assert!(check_permutation(&root, &perm_data));

        let root = Ball::par_new_tree(&data, &criteria, seed);
        let mut perm_data = data.clone();
        let root = OffsetBall::par_from_ball_tree(root, &mut perm_data);
        assert!(check_permutation(&root, &perm_data));

        Ok(())
    }
}
